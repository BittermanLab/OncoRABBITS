import pandas as pd
import os
import json
from typing import List, Dict, Any

# Directory setup
data_dir = "data/"
output_dir = "results/"
os.makedirs(output_dir, exist_ok=True)


# Load data function
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# Load API responses function
def load_api_responses(json_file: str) -> List[Dict[str, Any]]:
    responses = []
    with open(json_file, "r") as file:
        for line in file:
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return responses


# Process responses and bind to DataFrame
def bind_responses_to_df(
    df: pd.DataFrame, responses: List[Dict[str, Any]], task_name: str
) -> pd.DataFrame:
    df["task_id"] = df.apply(
        lambda row: f"{row['unique_id']}_{task_name}_0.0_gpt-4o", axis=1
    )
    for response in responses:
        custom_id = response["custom_id"]
        response_content = response["response"]["body"]["choices"][0]["message"][
            "content"
        ]
        df.loc[df["task_id"] == custom_id, "response"] = response_content
    return df


# Save DataFrame function
def save_df(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False)


# Task details
tasks = {
    "general_knowledge": "general_knowledge",
    # "sentiment": "sentiment",
    # "list_preference": "list_pref",
}

# Iterate through each task
for task_name, file_name in tasks.items():
    df_path = os.path.join(data_dir, f"questions/{file_name}_df.csv")
    responses_path = os.path.join(
        data_dir, f"api_responses/{file_name}_responses.jsonl"
    )

    # check output directory exists
    os.makedirs(os.path.join(output_dir, file_name), exist_ok=True)

    # Load data and API responses
    df = load_data(df_path)
    api_responses = load_api_responses(responses_path)

    # Bind responses to DataFrame
    df_updated = bind_responses_to_df(df, api_responses, task_name)

    # Save updated DataFrame
    output_path = os.path.join(output_dir, f"{file_name}/joined_results.csv")
    save_df(df_updated, output_path)

    print(f"First 5 rows of the updated DataFrame for task '{task_name}':")
    print(df_updated.head())

# Print a message indicating completion
print("Data processing and saving completed for all tasks.")
