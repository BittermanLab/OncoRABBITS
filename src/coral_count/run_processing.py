import os
import pandas as pd
import re
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import collections
import argparse

from datasets import load_dataset
from coral_count.drug_mapping import DrugMapper

# Adjust pandas display options
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", None)  # Show full column content
pd.set_option("display.width", 1000)  # Adjust the display width

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_dataframe(df):
    flat_records = [flatten_dict(record) for record in df.to_dict("records")]
    return pd.DataFrame(flat_records)


def contains_keyword(col_value, keywords):
    if col_value is None or (
        isinstance(col_value, (str, float)) and pd.isna(col_value)
    ):
        return False

    if isinstance(col_value, list):
        return any(
            re.search(rf"\b{re.escape(keyword)}\b", item, re.IGNORECASE)
            for item in col_value
            for keyword in keywords
        )

    if isinstance(col_value, str):
        return any(
            re.search(rf"\b{re.escape(keyword)}\b", col_value, re.IGNORECASE)
            for keyword in keywords
        )
    return False


def replace_drugs(prompt, old_keyword, new_keyword):
    prompt = str(prompt)
    old_keyword = str(old_keyword)
    new_keyword = str(new_keyword)

    pattern = re.compile(rf"\b{re.escape(old_keyword)}\b", re.IGNORECASE)
    replaced_prompt = pattern.sub(new_keyword, prompt)

    if prompt != replaced_prompt:
        logging.debug(
            f"Replaced '{old_keyword}' with '{new_keyword}' in '{prompt}' to get '{replaced_prompt}'"
        )
    return replaced_prompt


def replace_in_col(col_value, replacement_map):
    if col_value is None or (
        isinstance(col_value, (str, float)) and pd.isna(col_value)
    ):
        return col_value

    if isinstance(col_value, list):
        new_col_value = []
        for item in col_value:
            original_item = item
            for old_keyword, new_keyword in replacement_map.items():
                item = replace_drugs(item, old_keyword, new_keyword)
            if original_item != item:
                logging.debug(
                    f"Original list item: '{original_item}', Replaced list item: '{item}'"
                )
            new_col_value.append(item)
        return new_col_value

    if isinstance(col_value, str):
        original_value = col_value
        for old_keyword, new_keyword in replacement_map.items():
            col_value = replace_drugs(col_value, old_keyword, new_keyword)
        if original_value != col_value:
            logging.debug(
                f"Original value: '{original_value}', Replaced value: '{col_value}'"
            )
    return col_value


def extract_keywords(col_value, keywords):
    found_keywords = []
    if col_value is None or (
        isinstance(col_value, (str, float)) and pd.isna(col_value)
    ):
        return found_keywords

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


def process_batch(batch_data_df, cols_of_interest, replacement_map, keywords):
    logging.info(f"Processing batch with {len(batch_data_df)} rows.")

    # Initialize found_keywords column
    batch_data_df["found_keywords"] = batch_data_df[cols_of_interest].apply(
        lambda row: list(
            set(keyword for cell in row for keyword in extract_keywords(cell, keywords))
        ),
        axis=1,
    )

    logging.info(
        f"Keywords extracted for batch. Found keywords: {batch_data_df['found_keywords']}"
    )

    for col in cols_of_interest:
        logging.info(f"Replacing values in column: {col}")
        batch_data_df[col] = batch_data_df[col].apply(
            lambda x: replace_in_col(x, replacement_map)
        )
    logging.info("Replacement completed for batch.")
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

    modified_data = pd.DataFrame()
    futures = []
    num_batches = (total_rows + batch_size - 1) // batch_size

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=num_batches, desc="Submitting batches", unit="batch"
        ) as submit_progress:
            for start in range(0, total_rows, batch_size):
                end = start + batch_size
                batch_data_df = split_data_df.iloc[start:end]
                futures.append(
                    executor.submit(
                        process_batch,
                        batch_data_df,
                        cols_of_interest,
                        replacement_map,
                        keywords,
                    )
                )
                submit_progress.update(1)

        with tqdm(
            total=num_batches, desc="Collecting results", unit="batch"
        ) as collect_progress:
            for future in as_completed(futures):
                batch_result = future.result()
                logging.info(f"Batch processed with {len(batch_result)} rows.")
                modified_data = pd.concat([modified_data, batch_result])
                collect_progress.update(1)

    logging.info(f"Total rows after processing: {len(modified_data)}")
    return modified_data


def save_modified_dataset(
    data, transformation, split_name, output_dir, dataset_name, flatten=False
):
    output_path = os.path.join(
        output_dir, f"{dataset_name.replace('/', '_')}_{transformation}"
    )
    os.makedirs(output_path, exist_ok=True)

    df = pd.DataFrame(data)
    if flatten:
        df = flatten_dataframe(df)
    parquet_path = os.path.join(output_path, f"{split_name}.parquet")
    df.to_parquet(parquet_path, index=False)

    logging.info(
        f"Saved dataset for '{transformation}' split '{split_name}' to {parquet_path}"
    )

    df = pd.read_parquet(parquet_path)
    # logging.info(f"First 5 rows of saved dataset:\n{df.head()}")


def process_split_in_chunks(
    dataset_split,
    drug_mapper,
    brand_to_generic_map,
    generic_to_brand_map,
    cols_of_interest,
    output_folder,
    dataset_name,
    max_workers=4,
):
    dataset_split_df = dataset_split.to_pandas()

    # Create unique id for each row
    dataset_split_df["local_id"] = dataset_split_df.index

    keywords = drug_mapper.load_all_keywords_list()
    logging.info("Extracting keywords from the original dataset...")

    # Extract keywords and filter rows with no keywords found
    dataset_split_df["found_keywords"] = dataset_split_df[cols_of_interest].apply(
        lambda row: list(
            set(keyword for cell in row for keyword in extract_keywords(cell, keywords))
        ),
        axis=1,
    )
    logging.info(
        f"Keyword extraction completed. Keywords found in {dataset_split_df['found_keywords'].apply(len).sum()} cells."
    )
    logging.info(f"First 5 rows after keyword extraction:\n{dataset_split_df.head()}")

    filtered_data_original = dataset_split_df[
        dataset_split_df["found_keywords"].apply(len) > 0
    ].copy()
    logging.info(f"Filtered original data contains {len(filtered_data_original)} rows.")

    # Ensure the local_id is maintained correctly
    filtered_data_original = filtered_data_original.reset_index(drop=True)

    # Process brand to generic replacements
    logging.info("Processing brand to generic replacements...")
    modified_data_btog = modify_dataset_parallel(
        filtered_data_original.copy(),
        cols_of_interest,
        brand_to_generic_map,
        keywords,
        max_workers=max_workers,
    )

    # Process generic to brand replacements
    logging.info("Processing generic to brand replacements...")
    modified_data_gtob = modify_dataset_parallel(
        filtered_data_original.copy(),
        cols_of_interest,
        generic_to_brand_map,
        keywords,
        max_workers=max_workers,
    )

    # Ensure all dataframes have the same local_id
    modified_data_btog["local_id"] = filtered_data_original["local_id"]
    modified_data_gtob["local_id"] = filtered_data_original["local_id"]

    logging.info(f"Filtered data (original): {len(filtered_data_original)} rows.")
    logging.info(f"Filtered data (brand to generic): {len(modified_data_btog)} rows.")
    logging.info(f"Filtered data (generic to brand): {len(modified_data_gtob)} rows.")

    flatten = "bigbio/pubmed_qa" in dataset_name

    logging.info("Saving original filtered dataset...")
    save_modified_dataset(
        filtered_data_original.to_dict("records"),
        "original_filtered",
        "test",
        output_folder,
        dataset_name,
        flatten=flatten,
    )

    logging.info("Saving brand to generic filtered dataset...")
    save_modified_dataset(
        modified_data_btog.to_dict("records"),
        "brand_to_generic_filtered",
        "test",
        output_folder,
        dataset_name,
        flatten=flatten,
    )

    logging.info("Saving generic to brand filtered dataset...")
    save_modified_dataset(
        modified_data_gtob.to_dict("records"),
        "generic_to_brand_filtered",
        "test",
        output_folder,
        dataset_name,
        flatten=flatten,
    )


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mapper = DrugMapper(args.brand_to_generic_csv_path, args.generic_to_brand_csv_path)
    brand_to_generic_map = mapper.load_keywords("brand_to_generic")
    generic_to_brand_map = mapper.load_keywords("generic_to_brand")

    # logging.info(f"Loaded brand to generic map: {brand_to_generic_map}")
    # logging.info(f"Loaded generic to brand map: {generic_to_brand_map}")

    dataset = pd.read_csv(args.dataset_path)
    dataset_name = args.dataset_name

    cols_of_interest = ["note_text"]

    dataset_output_dir = os.path.join(args.output_dir, dataset_name.replace("/", "_"))
    os.makedirs(dataset_output_dir, exist_ok=True)

    output_folder = os.path.join(dataset_output_dir, "test")
    os.makedirs(output_folder, exist_ok=True)

    process_split_in_chunks(
        dataset,
        mapper,
        brand_to_generic_map,
        generic_to_brand_map,
        cols_of_interest,
        output_folder,
        dataset_name,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets with drug name mappings."
    )
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
        default="pre_filter_datasets",
        help="Directory to save the processed datasets.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coral",
        help="Specific dataset to process or 'all' for processing all datasets.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of worker processes to use for parallel processing.",
    )
    args = parser.parse_args()

    main(args)
