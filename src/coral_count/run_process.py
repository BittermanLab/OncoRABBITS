import os
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to replace keywords in a prompt
def replace_drugs(prompt, old_keyword, new_keyword):
    prompt = str(prompt)
    old_keyword = str(old_keyword)
    new_keyword = str(new_keyword)

    pattern = re.compile(rf"\b{re.escape(old_keyword)}\b", re.IGNORECASE)
    prompt = pattern.sub(new_keyword, prompt)
    return prompt


# Function to replace keywords in a column value
def replace_in_col(col_value, replacement_map):
    if col_value is None or (
        isinstance(col_value, (str, float)) and pd.isna(col_value)
    ):
        return col_value

    if isinstance(col_value, list):
        return [
            replace_drugs(item, old_keyword, new_keyword)
            for item in col_value
            for old_keyword, new_keyword in replacement_map.items()
        ]

    if isinstance(col_value, str):
        for old_keyword, new_keyword in replacement_map.items():
            col_value = replace_drugs(col_value, old_keyword, new_keyword)
    return col_value


# Function to check if a keyword is present in a column value
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


# Function to extract keywords from a column value
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

    return list(set(found_keywords))


# Function to process a batch and replace keywords
def process_batch(batch_data_df, cols_of_interest, replacement_map, keywords):
    batch_data_df["found_keywords"] = batch_data_df[cols_of_interest].apply(
        lambda row: list(
            set(
                [
                    keyword
                    for cell in row
                    for keyword in extract_keywords(cell, keywords)
                ]
            )
        ),
        axis=1,
    )

    for col in cols_of_interest:
        batch_data_df[col] = batch_data_df[col].apply(
            lambda x: replace_in_col(x, replacement_map)
        )
    return batch_data_df.to_dict("records")


# Function to replace keywords in parallel and write out results for each batch
def replace_keywords_parallel(
    split_data, cols_of_interest, replacement_map, keywords, max_workers=4
):
    split_data_df = pd.DataFrame(split_data)
    modified_data = []

    total_rows = len(split_data_df)
    batch_size = (total_rows + max_workers - 1) // max_workers
    logging.info(
        f"Processing {total_rows} rows in {max_workers} batches of size {batch_size}"
    )

    futures = []
    num_batches = (total_rows + batch_size - 1) // batch_size

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=num_batches, desc=f"Submitting batches", unit="batch"
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
            total=num_batches, desc=f"Collecting results", unit="batch"
        ) as collect_progress:
            for future in as_completed(futures):
                modified_data.extend(future.result())
                collect_progress.update(1)

    return modified_data


# Function to save modified dataset
def save_modified_dataset(data, transformation, output_dir, dataset_name, split_name):
    output_path = os.path.join(
        output_dir, f"{dataset_name.replace('/', '_')}_{transformation}"
    )
    os.makedirs(output_path, exist_ok=True)

    df = pd.DataFrame(data)
    parquet_path = os.path.join(output_path, f"{split_name}.parquet")
    df.to_parquet(parquet_path, index=False)

    logging.info(
        f"Saved dataset for '{transformation}' split '{split_name}' to {parquet_path}"
    )

    df = pd.read_parquet(parquet_path)
    logging.info(df.head(5))


# Main function to run the processing
def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load the keyword mappings
    logging.info("Loading keyword mappings...")
    brand_to_generic_map = (
        pd.read_csv(args.brand_to_generic_csv_path)
        .set_index("brand")["generic"]
        .to_dict()
    )
    generic_to_brand_map = (
        pd.read_csv(args.generic_to_brand_csv_path)
        .set_index("generic")["brand"]
        .to_dict()
    )

    keywords = list(brand_to_generic_map.keys()) + list(generic_to_brand_map.keys())

    # Load the dataset
    logging.info("Loading dataset...")
    dataset = pd.read_csv(args.dataset_path)

    cols_of_interest = ["note_text"]

    if args.debug:
        dataset = dataset.head(5)

    logging.info(f"Total rows in dataset: {len(dataset)}")

    dataset_output_dir = os.path.join(args.output_dir, "coral/pre_edit")
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)

    # Add found_keywords to the original dataset
    logging.info("Extracting keywords from the original dataset...")
    dataset["found_keywords"] = dataset[cols_of_interest].apply(
        lambda row: list(
            set(
                [
                    keyword
                    for cell in row
                    for keyword in extract_keywords(cell, keywords)
                ]
            )
        ),
        axis=1,
    )

    # Process brand to generic replacements
    logging.info("Processing brand to generic replacements...")
    modified_data_btog = replace_keywords_parallel(
        dataset,
        cols_of_interest,
        brand_to_generic_map,
        keywords,
        max_workers=args.max_workers,
    )

    # Process generic to brand replacements
    logging.info("Processing generic to brand replacements...")
    modified_data_gtob = replace_keywords_parallel(
        dataset,
        cols_of_interest,
        generic_to_brand_map,
        keywords,
        max_workers=args.max_workers,
    )

    # Filter to keep only rows with found_keywords in any dataset
    logging.info("Filtering datasets to keep only rows with found keywords...")
    filtered_data_original = dataset[dataset["found_keywords"].apply(len) > 0]

    # Create a set of indices of rows with keywords in btog and gtob
    indices_with_keywords = set(filtered_data_original.index)

    filtered_data_btog = pd.DataFrame(modified_data_btog)
    filtered_data_gtob = pd.DataFrame(modified_data_gtob)

    filtered_data_btog["found_keywords"] = dataset["found_keywords"]
    filtered_data_gtob["found_keywords"] = dataset["found_keywords"]

    # Ensure all three datasets contain the same rows
    filtered_data_original = dataset.loc[list(indices_with_keywords)]
    filtered_data_btog = filtered_data_btog.loc[list(indices_with_keywords)]
    filtered_data_gtob = filtered_data_gtob.loc[list(indices_with_keywords)]

    # Save the datasets
    logging.info("Saving original filtered dataset...")
    save_modified_dataset(
        filtered_data_original.to_dict("records"),
        "original_filtered",
        dataset_output_dir,
        args.dataset_name,
        "all",
    )

    logging.info("Saving brand to generic filtered dataset...")
    save_modified_dataset(
        filtered_data_btog.to_dict("records"),
        "brand_to_generic_filtered",
        dataset_output_dir,
        args.dataset_name,
        "all",
    )

    logging.info("Saving generic to brand filtered dataset...")
    save_modified_dataset(
        filtered_data_gtob.to_dict("records"),
        "generic_to_brand_filtered",
        dataset_output_dir,
        args.dataset_name,
        "all",
    )


if __name__ == "__main__":
    import argparse

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
        "--dataset_name",
        type=str,
        default="all",
        help="Specific dataset to process or 'all' for processing all datasets.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=30,
        help="Maximum number of worker processes to use for parallel processing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with a small subset of data.",
    )
    args = parser.parse_args()

    main(args)
