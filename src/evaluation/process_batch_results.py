import pandas as pd
import os
import json
from typing import List, Dict, Any

from list_utils import process_list_preference
from general_knowledge_utils import process_general_knowledge
from sentiment_utils import process_sentiment

# Directory setup
data_dir = "data/"
output_dir = "results/"
os.makedirs(output_dir, exist_ok=True)


# Load data function
def load_data(file_path: str) -> pd.DataFrame:
    # if contains list -> remove before first _
    if "list" in file_path:
        file_path = "data/questions/list_preference_df.csv"
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

    temperatures = ["0.0", "0.7", "1.0"]

    # Initialize response columns for each temperature
    for temp in temperatures:
        df[f"response_{temp}"] = ""

    for response in responses:
        custom_id = response["custom_id"]
        response_content = response["response"]["body"]["choices"][0]["message"][
            "content"
        ]
        # # Extract the temperature from the custom_id
        parts = custom_id.rsplit("_", 2)
        temp = parts[1]

        if temp in temperatures:
            df.loc[
                df["unique_id"].apply(lambda x: f"{x}_{task_name}_{temp}_gpt-4o")
                == custom_id,
                f"response_{temp}",
            ] = response_content

    return df


# Save DataFrame function
def save_df(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False)


# Task details
tasks = {
    "general_knowledge": "general_knowledge",
    "sentiment_question_about": "sentiment",
    "sentiment_question_patient": "sentiment",
    "sentiment_question_physician": "sentiment",
    "list_preference_prompt1": "list_preference",
    "list_preference_prompt2": "list_preference",
}


# Iterate through each task
for task_name, file_name in tasks.items():

    df_path = os.path.join(data_dir, f"questions/{file_name}_df.csv")
    responses_path = os.path.join(
        data_dir, f"api_responses/{task_name}_responses.jsonl"
    )

    # check output directory exists
    os.makedirs(os.path.join(output_dir, file_name), exist_ok=True)

    # Load data and API responses
    df = load_data(df_path)
    api_responses = load_api_responses(responses_path)

    # Bind responses to DataFrame
    df_updated = bind_responses_to_df(df, api_responses, task_name)

    print(f"First row of the updated DataFrame for task '{task_name}':")
    print(df_updated.head(1))
    print(df_updated.columns)

    # unique task evaluation
    if task_name == "list_preference_prompt1" or task_name == "list_preference_prompt2":
        results_df = process_list_preference(df_updated, output_dir, task_name)
    elif (
        task_name == "sentiment_question_about"
        or task_name == "sentiment_question_patient"
        or task_name == "sentiment_question_physician"
    ):
        results_df = process_sentiment(df_updated, output_dir, task_name)
    elif task_name == "general_knowledge":
        results_df = process_general_knowledge(df_updated, output_dir)
    else:
        print("Error: Task not found")

    print(f"{task_name} processing completed.")
    print("\n" * 5)

# Print a message indicating completion
print("Data processing and saving completed for all tasks.")