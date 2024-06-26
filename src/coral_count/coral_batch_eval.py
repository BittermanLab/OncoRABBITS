import os
import pandas as pd
import json
from collections import Counter
import re
from tqdm.auto import tqdm
import logging


def load_and_process_data(data_dir):
    """
    Function to load and process the dataset from the specified directory.
    """
    file_path = os.path.join(data_dir, "combined_notes.csv")
    df = pd.read_csv(file_path)
    # debug - take first 5 rows
    df = df.head()
    return df


def load_api_responses(file_path):
    """
    Function to load API responses from a JSONL file.
    """
    # debug - read only 5 lines
    with open(file_path, "r") as file:
        responses = [json.loads(file.readline()) for _ in range(5)]
    return responses


def bind_responses_to_df(df, responses):
    """
    Function to bind API responses to the DataFrame based on the 'coral_idx' field.
    """
    # Extract the custom_id and the relevant response content
    response_dict = {
        response["custom_id"].split("_")[0]: response["response"]["body"]["choices"][0][
            "message"
        ]["content"]
        for response in responses
    }
    df["response"] = df["coral_idx"].astype(str).map(response_dict)
    # add custom_id to df
    df["custom_id"] = (
        df["coral_idx"]
        .astype(str)
        .map(
            {
                response["custom_id"].split("_")[0]: response["custom_id"]
                for response in responses
            }
        )
    )
    return df


def custom_parse_response(response):
    """
    Custom function to parse drug list from response string, with improved handling
    for control characters.
    """
    if isinstance(response, str):
        try:
            # Remove leading and trailing square brackets
            response = response.strip("[]")
            # Split the response by commas not within quotes
            parts = re.split(r'(?<!"),(?!")', response)
            # Strip whitespace and add double quotes around parts if not already quoted
            parts = [part.strip().strip('"') for part in parts]
            # Properly escape control characters in each part
            parts = [json.dumps(part) for part in parts]
            # Join parts back with commas
            response = ",".join(parts)
            # Form the final JSON array
            response = f"[{response}]"
            # Parse the JSON
            drug_list = json.loads(response)
            return drug_list
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing response: {response}, error: {e}")
            return []
    else:
        return []


def extract_keywords_from_col(df, col, keywords):
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
                    if re.search(
                        rf"\b{re.escape(keyword)}\b", str(item), re.IGNORECASE
                    ):
                        found_keywords.append(keyword)
        elif isinstance(col_value, str):
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", col_value, re.IGNORECASE):
                    found_keywords.append(keyword)

        logging.debug(f"Extracted keywords from '{col_value}': {found_keywords}")
        return list(set(found_keywords))

    new_col_name = f"{col}_keywords"
    df[new_col_name] = df[col].apply(
        lambda x: extract_keywords(custom_parse_response(x), keywords)
    )

    keyword_counts = Counter()
    for keywords_list in df[new_col_name]:
        keyword_counts.update(keywords_list)

    return keyword_counts


def aggregate_keyword_counts(counts1, counts2):
    """
    Aggregates keyword counts from two sources, imputing 0 for keywords not found in both.

    Parameters:
    - counts1: A Counter object with keyword counts from the first source (response).
    - counts2: A Counter object with keyword counts from the second source (note_text).

    Returns:
    A DataFrame with the aggregated keyword counts.
    """

    # Convert Counter objects to DataFrames
    df1 = (
        pd.DataFrame.from_dict(counts1, orient="index", columns=["response_count"])
        .reset_index()
        .rename(columns={"index": "Keyword"})
    )
    df2 = (
        pd.DataFrame.from_dict(counts2, orient="index", columns=["note_text_count"])
        .reset_index()
        .rename(columns={"index": "Keyword"})
    )

    # Merge the DataFrames on Keyword, imputing 0 where counts are missing
    merged_df = pd.merge(df1, df2, on="Keyword", how="outer").fillna(0)

    # Convert counts back to integers (they were converted to floats by fillna)
    merged_df["response_count"] = merged_df["response_count"].astype(int)
    merged_df["note_text_count"] = merged_df["note_text_count"].astype(int)

    return merged_df


if __name__ == "__main__":
    model = "gpt-4o"
    data_dir = "src/coral_count/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/unannotated/data"
    out_dir = "data/coral/counts"
    responses_path = os.path.join("data", "api_responses", model, "coral_summary.jsonl")

    # Load and process the dataset
    combined_df = load_and_process_data(data_dir)
    print("Loaded and processed dataset:")
    print(combined_df.head())

    # Load API responses
    api_responses = load_api_responses(responses_path)
    print(f"Loaded {len(api_responses)} API responses")

    # Bind responses to the DataFrame
    df_with_responses = bind_responses_to_df(combined_df, api_responses)
    print("DataFrame with responses:")
    print(df_with_responses.head())

    # Load all keywords
    from drug_mapping import DrugMapper

    mapper = DrugMapper(
        "data/drug_names/brand_to_generic_df.csv",
        "data/drug_names/generic_to_brand_df.csv",
    )
    all_keywords = mapper.load_all_keywords_list()

    cols_of_interest = ["note_text", "response"]

    # Extract keyword counts from each column of interest
    response_counts = extract_keywords_from_col(
        df_with_responses, "response", all_keywords
    )
    note_text_counts = extract_keywords_from_col(
        df_with_responses, "note_text", all_keywords
    )

    # Aggregate the counts
    drug_counts = aggregate_keyword_counts(response_counts, note_text_counts)

    # debug - print drug counts
    print(response_counts)
    print("-" * 50)
    print(note_text_counts)
    print("-" * 50)
    print(drug_counts)

    # Save the count to a CSV file
    drug_counts_file = os.path.join(out_dir, "drug_counts.csv")
    drug_counts.to_csv(drug_counts_file, index=False)
    print(f"Saved drug counts to {drug_counts_file}")

    # Optional: Save the resulting DataFrame to a CSV file
    output_file = os.path.join(out_dir, "combined_df_with_responses.csv")
    df_with_responses.to_csv(output_file, index=False)
    print(f"Saved DataFrame with responses to {output_file}")
