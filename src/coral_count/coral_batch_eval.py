import os
import pandas as pd
import json


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


if __name__ == "__main__":
    data_dir = "src/coral_count/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/unannotated/data"
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

    # Optional: Save the resulting DataFrame to a CSV file
    output_file = os.path.join(data_dir, "combined_df_with_responses.csv")
    df_with_responses.to_csv(output_file, index=False)
    print(f"Saved DataFrame with responses to {output_file}")
