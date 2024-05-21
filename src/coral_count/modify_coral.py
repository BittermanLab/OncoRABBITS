import os
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
MAX_BATCH_SIZE = 2500  # Set the maximum batch size


# Function to count keywords in a prompt
def count_keywords(prompt, keywords):
    prompt = str(prompt)
    keyword_counts = {keyword: 0 for keyword in keywords}
    for keyword in keywords:
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        keyword_counts[keyword] = len(pattern.findall(prompt))
    return keyword_counts


# Function to count keywords in a column value
def count_in_col(col_value, keywords):
    if col_value is None or (
        isinstance(col_value, (str, float)) and pd.isna(col_value)
    ):
        return {keyword: 0 for keyword in keywords}

    if isinstance(col_value, list):
        keyword_counts = {keyword: 0 for keyword in keywords}
        for item in col_value:
            item_counts = count_keywords(item, keywords)
            for keyword, count in item_counts.items():
                keyword_counts[keyword] += count
        return keyword_counts

    if isinstance(col_value, str):
        return count_keywords(col_value, keywords)

    return {keyword: 0 for keyword in keywords}


# Function to process a batch and write the result to a file
def process_batch(
    batch_data_df,
    cols_of_interest,
    keywords,
    split_name,
    keyword_type,
    batch_index,
    output_dir,
):
    keyword_counts = {keyword: 0 for keyword in keywords}
    for col in cols_of_interest:
        col_counts = batch_data_df[col].apply(lambda x: count_in_col(x, keywords))
        for count_dict in col_counts:
            for keyword, count in count_dict.items():
                keyword_counts[keyword] += count

    # Write the result to a file
    batch_file = os.path.join(
        output_dir, f"{split_name}_{keyword_type}_batch_{batch_index}.parquet"
    )
    df = pd.DataFrame(list(keyword_counts.items()), columns=["keyword", "count"])
    df.to_parquet(batch_file, index=False)
    logging.info(
        f"Processed and saved batch {batch_index} for split {split_name} and keyword type {keyword_type}"
    )

    # Clear memory
    del batch_data_df, col_counts, df, keyword_counts
    gc.collect()


# Function to split data into batches
def split_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data.iloc[i : i + batch_size]


# Function to count keywords in parallel and write out results for each batch
def count_keywords_parallel(
    split_data,
    cols_of_interest,
    keywords,
    split_name,
    keyword_type,
    output_dir,
    max_workers=4,
):
    split_data_df = pd.DataFrame(split_data)
    total_rows = len(split_data_df)

    # Determine batch size based on the maximum batch size
    batch_size = min((total_rows + max_workers - 1) // max_workers, MAX_BATCH_SIZE)
    num_batches = (total_rows + batch_size - 1) // batch_size

    futures = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=num_batches,
            desc=f"Submitting batches for {split_name} ({keyword_type})",
            unit="batch",
        ) as submit_progress:
            for batch_index, start in enumerate(range(0, total_rows, batch_size)):
                end = start + batch_size
                batch_data_df = split_data_df.iloc[start:end]
                futures.append(
                    executor.submit(
                        process_batch,
                        batch_data_df,
                        cols_of_interest,
                        keywords,
                        split_name,
                        keyword_type,
                        batch_index,
                        output_dir,
                    )
                )
                submit_progress.update(1)

        with tqdm(
            total=num_batches,
            desc=f"Collecting results for {split_name} ({keyword_type})",
            unit="batch",
        ) as collect_progress:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(
                        f"Error collecting results for batch {futures.index(future)}: {e}"
                    )
                collect_progress.update(1)

    # Clear memory
    del split_data_df
    gc.collect()


# Function to aggregate results from all batch files
def aggregate_results(output_dir, split_name, keyword_type, final_output_file):
    batch_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(f"{split_name}_{keyword_type}_batch_")
    ]

    # Debug: Log the batch files found
    logging.info(f"Batch files found for {split_name} ({keyword_type}): {batch_files}")

    if not batch_files:
        logging.error(
            f"No batch files found for split {split_name} and keyword type {keyword_type} in directory {output_dir}"
        )
        return

    all_batches = [pd.read_parquet(batch_file) for batch_file in batch_files]
    final_result = pd.concat(all_batches, ignore_index=True)
    final_result = final_result.groupby("keyword").sum().reset_index()
    final_result.to_parquet(final_output_file, index=False)
    logging.info(f"Aggregated and saved final results to {final_output_file}")

    # Clear memory
    del all_batches, final_result
    gc.collect()


# Main function to run the processing
def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    brand_to_generic_keywords = pd.read_csv(args.brand_to_generic_csv_path)[
        "brand"
    ].tolist()
    generic_to_brand_keywords = pd.read_csv(args.generic_to_brand_csv_path)[
        "generic"
    ].tolist()

    hf_datasets = {
        # "medmcqa": ["question", "opa", "opb", "opc", "opd"],
        # "bigbio/pubmed_qa": ["CONTEXTS", "QUESTION"],
        # "GBaker/MedQA-USMLE-4-options-hf": [
        #     "sent1",
        #     "ending0",
        #     "ending1",
        #     "ending2",
        #     "ending3",
        # ],
        # "augtoma/usmle_step_1": ["question", "options"],
        # "augtoma/usmle_step_2": ["question", "options"],
        # "augtoma/usmle_step_3": ["question", "options"],
        # "hails/mmlu_no_train/anatomy": ["question", "choices"],
        # "hails/mmlu_no_train/clinical_knowledge": ["question", "choices"],
        # "hails/mmlu_no_train/college_medicine": ["question", "choices"],
        # "hails/mmlu_no_train/medical_genetics": ["question", "choices"],
        # "hails/mmlu_no_train/professional_medicine": ["question", "choices"],
        # "hails/mmlu_no_train/college_biology": ["question", "choices"],
    }

    datasets_to_process = (
        [args.dataset_name]
        if args.dataset_name.lower() != "all"
        else hf_datasets.keys()
    )

    for dataset_name in tqdm(datasets_to_process, desc="Overall Progress"):
        print(f"Processing dataset: {dataset_name}")
        if dataset_name in hf_datasets:
            cols_of_interest = hf_datasets[dataset_name]
            # if dataset_name contains hails-> we need to split hails/mmlu_no_train/anatomy into load-dataset(hails/mmlu_no_train, anatomy)
            if "hails" in dataset_name:
                local_dataset_name = dataset_name.split("/")
                dataset = load_dataset("hails/mmlu_no_train", local_dataset_name[2])
            else:
                dataset = load_dataset(dataset_name)
            dataset_output_dir = os.path.join(
                args.output_dir, dataset_name.replace("/", "_")
            )
            if not os.path.exists(dataset_output_dir):
                os.makedirs(dataset_output_dir)

            for split in dataset.keys():
                output_folder = os.path.join(dataset_output_dir, split)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Process brand to generic keywords
                count_keywords_parallel(
                    dataset[split],
                    cols_of_interest,
                    brand_to_generic_keywords,
                    split_name=split,
                    keyword_type="brand_to_generic",
                    output_dir=output_folder,
                    max_workers=args.max_workers,
                )

                # Process generic to brand keywords
                count_keywords_parallel(
                    dataset[split],
                    cols_of_interest,
                    generic_to_brand_keywords,
                    split_name=split,
                    keyword_type="generic_to_brand",
                    output_dir=output_folder,
                    max_workers=args.max_workers,
                )

                # Aggregate results and save
                brand_to_generic_file = os.path.join(
                    output_folder, f"{split}_brand_to_generic.parquet"
                )
                generic_to_brand_file = os.path.join(
                    output_folder, f"{split}_generic_to_brand.parquet"
                )

                aggregate_results(
                    output_folder, split, "brand_to_generic", brand_to_generic_file
                )
                aggregate_results(
                    output_folder, split, "generic_to_brand", generic_to_brand_file
                )

                # Clear memory
                gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count keywords in datasets.")
    parser.add_argument(
        "--brand_to_generic_csv_path",
        type=str,
        default="RxNorm_eval/brand_to_generic.csv",
        help="Path to the CSV file containing brand to generic drug mappings.",
    )
    parser.add_argument(
        "--generic_to_brand_csv_path",
        type=str,
        default="RxNorm_eval/generic_to_brand.csv",
        help="Path to the CSV file containing generic to brand drug mappings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="counts",
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
    args = parser.parse_args()

    main(args)

    # Check the aggregation
    final_output_file = "counts/augtoma_usmle_step_1/test/test_generic_to_brand.parquet"

    df = pd.read_parquet(final_output_file)
    print(df.head())
