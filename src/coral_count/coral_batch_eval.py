import os
import pandas as pd
import json
from collections import Counter
import re


def load_and_process_data(data_dir):
    """
    Function to load and process the dataset from the specified directory.
    """
    file_path = os.path.join(data_dir, "combined_notes.csv")
    df = pd.read_csv(file_path)
    return df


def load_api_responses(file_path):
    """
    Function to load API responses from a JSONL file.
    """
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


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
    return df


def custom_parse_response(response):
    """
    Custom function to parse drug list from response string.
    """
    if isinstance(response, str):
        try:
            # Remove leading and trailing square brackets
            response = response.strip("[]")
            # Split the response by commas not within quotes
            parts = re.split(r'(?<!"),(?!")', response)
            # Strip whitespace and add double quotes around parts if not already quoted
            parts = [part.strip().strip('"') for part in parts]
            # Join parts back with quotes and commas
            response = ",".join([f'"{part}"' for part in parts])
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


def count_drug_mentions(df):
    """
    Function to count the sum of each drug mention in the response column.
    """
    df["drugs"] = df["response"].apply(custom_parse_response)
    all_drugs = [drug for drugs in df["drugs"] for drug in drugs]
    drug_counts = Counter(all_drugs)

    # Convert counts to a DataFrame
    drug_counts_df = pd.DataFrame.from_dict(
        drug_counts, orient="index", columns=["count"]
    )
    drug_counts_df = drug_counts_df.reset_index().rename(columns={"index": "drug"})

    return drug_counts_df


if __name__ == "__main__":
    data_dir = "src/coral_count/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/unannotated/data"
    out_dir = "data/coral"
    responses_path = os.path.join("data", "api_responses", "coral_responses.jsonl")

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

    # Count drug mentions
    drug_counts_df = count_drug_mentions(df_with_responses)
    print("Drug mentions counted:")
    drug_counts_df = drug_counts_df.sort_values(by="count", ascending=False)
    print(drug_counts_df.head())

    # Save the count to a CSV file
    drug_counts_file = os.path.join(out_dir, "drug_counts.csv")
    drug_counts_df.to_csv(drug_counts_file, index=False)
    print(f"Saved drug counts to {drug_counts_file}")

    # Optional: Save the resulting DataFrame to a CSV file
    output_file = os.path.join(out_dir, "gpt_batch/combined_df_with_responses.csv")
    df_with_responses.to_csv(output_file, index=False)
    print(f"Saved DataFrame with responses to {output_file}")
