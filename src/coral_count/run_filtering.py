import os
import pandas as pd
import re
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import collections
import argparse
from datasets import load_dataset
from drug_mapping import DrugMapper

# Adjust pandas display options
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", None)  # Show full column content
pd.set_option("display.width", 1000)  # Adjust the display width

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_keywords(col_value, keywords):
    found_keywords = []
    if col_value is None or (
        isinstance(col_value, (str, float)) and pd.isna(col_value)
    ):
        return found_keywords

    keywords = sorted(keywords, key=len, reverse=True)

    if isinstance(col_value, list):
        for item in col_value:
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", item, re.IGNORECASE):
                    found_keywords.append(keyword)

    if isinstance(col_value, str):
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", col_value, re.IGNORECASE):
                found_keywords.append(keyword)

    logging.debug(f"Extracted keywords from '{col_value}': {found_keywords}")
    return list(set(found_keywords))


def extract_keywords_batch(batch_data_df, cols_of_interest, keywords):
    logging.info(f"Extracting keywords for batch with {len(batch_data_df)} rows.")

    batch_data_df["found_keywords"] = batch_data_df[cols_of_interest].apply(
        lambda row: list(
            set(keyword for cell in row for keyword in extract_keywords(cell, keywords))
        ),
        axis=1,
    )

    logging.info(
        f"Keywords extracted for batch. Found keywords: {batch_data_df['found_keywords'].apply(len).sum()}"
    )
    return batch_data_df


def modify_dataset_parallel(
    split_data_df, cols_of_interest, replacement_map, keywords, max_workers=4
):
    total_rows = len(split_data_df)
    if total_rows == 0:
        logging.warning("Empty dataset provided.")
        return split_data_df

    batch_size = (total_rows + max_workers - 1) // max_workers
    logging.info(
        f"Processing {total_rows} rows in {max_workers} batches of size {batch_size}"
    )

    futures = []
    num_batches = (total_rows + batch_size - 1) // batch_size

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=num_batches,
            desc="Submitting keyword extraction batches",
            unit="batch",
        ) as submit_progress:
            for start in range(0, total_rows, batch_size):
                end = start + batch_size
                batch_data_df = split_data_df.iloc[start:end]
                futures.append(
                    executor.submit(
                        extract_keywords_batch,
                        batch_data_df,
                        cols_of_interest,
                        keywords,
                    )
                )
                submit_progress.update(1)

        modified_data = pd.DataFrame()
        with tqdm(
            total=num_batches,
            desc="Collecting keyword extraction results",
            unit="batch",
        ) as collect_progress:
            for future in as_completed(futures):
                batch_result = future.result()
                logging.info(f"Batch processed with {len(batch_result)} rows.")
                modified_data = pd.concat([modified_data, batch_result])
                collect_progress.update(1)

    return modified_data


def process_split_in_chunks(
    dataset_split_df,
    drug_mapper,
    brand_to_generic_map,
    generic_to_brand_map,
    cols_of_interest,
    output_folder,
    dataset_name,
    split_name,
    max_workers=4,
):
    # Create unique id for each row
    dataset_split_df["local_id"] = dataset_split_df.index

    keywords = drug_mapper.load_all_keywords_list()

    # Extract keywords and filter rows with no keywords found
    extracted_data = modify_dataset_parallel(
        dataset_split_df,
        cols_of_interest,
        None,  # No replacement map for extraction
        keywords,
        max_workers=max_workers,
    )

    filtered_data_original = extracted_data[
        extracted_data["found_keywords"].apply(len) > 0
    ].copy()
    logging.info(
        f"Keyword extraction completed. Keywords found in {filtered_data_original['found_keywords'].apply(len).sum()} cells."
    )

    # Ensure the local_id is maintained correctly
    filtered_data_original = filtered_data_original.reset_index(drop=True)

    # Save the filtered dataset to a parquet file
    filtered_parquet_path = os.path.join(
        output_folder, f"{dataset_name.replace('/', '_')}_{split_name}_filtered.parquet"
    )
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filtered_parquet_path), exist_ok=True)

    filtered_data_original.to_parquet(filtered_parquet_path, index=False)
    logging.info(
        f"Filtered original data contains {len(filtered_data_original)} rows. Saved to {filtered_parquet_path}"
    )


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mapper = DrugMapper(args.brand_to_generic_csv_path, args.generic_to_brand_csv_path)
    brand_to_generic_map = mapper.load_keywords("brand_to_generic")
    generic_to_brand_map = mapper.load_keywords("generic_to_brand")

    logging.info(f"Loaded brand to generic map: {brand_to_generic_map}")
    logging.info(f"Loaded generic to brand map: {generic_to_brand_map}")

    dataset = pd.read_csv(args.dataset_path)

    cols_of_interest = ["note_text"]
    split = "all"
    dataset_name = "coral"

    output_folder = os.path.join(args.output_dir, "coral/pre_edit")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_split_in_chunks(
        dataset,
        mapper,
        brand_to_generic_map,
        generic_to_brand_map,
        cols_of_interest,
        output_folder,
        dataset_name,
        split,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace drug names in datasets.")
    parser.add_argument(
        "--brand_to_generic_csv_path",
        type=str,
        default="data/drug_names/brand_to_generic_df.csv",
        help="Path to the CSV file containing brand to generic drug mappings.",
    )
    parser.add_argument(
        "--generic_to_brand_csv_path",
        type=str,
        default="data/drug_names/generic_to_brand_df.csv",
        help="Path to the CSV file containing generic to brand drug mappings.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="src/coral_count/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/unannotated/data/combined_notes.csv",
        help="Path to the dataset to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save the processed datasets.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of worker processes to use for parallel processing.",
    )
    args = parser.parse_args()

    main(args)
