import pandas as pd
import os
import json
from typing import List, Dict, Any

from sentiment_utils import process_sentiment
from other_utils import save_statistics

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# Directory setup
data_dir = "data/"
output_dir = "results/"
os.makedirs(output_dir, exist_ok=True)


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
    responses: List[Dict[str, Any]], task_name: str, model_name: str, type_: str
) -> pd.DataFrame:
    temperatures = ["0.0", "0.7", "1.0"]

    # Initialize a dictionary to hold lists of responses by unique_id and type
    df_dict = {"unique_id": [], "type": []}
    for temp in temperatures:
        df_dict[f"response_{temp}"] = []

    for response in responses:
        custom_id = response["custom_id"]
        response_content = response["response"]["body"]["choices"][0]["message"][
            "content"
        ]
        parts = custom_id.rsplit("_", 5)
        unique_id = parts[0]
        temp = parts[-2]

        if temp in temperatures:
            df_dict["unique_id"].append(unique_id)
            df_dict["type"].append(type_)
            for t in temperatures:
                if t == temp:
                    df_dict[f"response_{t}"].append(response_content)
                else:
                    df_dict[f"response_{t}"].append(None)

    df = pd.DataFrame(df_dict)

    # Combine rows with the same unique_id and type
    df = df.groupby(["unique_id", "type"]).first().reset_index()

    return df


# Define models, types, and tasks
models = ["gpt-3.5-turbo-0125", "gpt-4o", "gpt-4-turbo"]
types = ["brand", "generic"]
# tasks = ["coral_sentiment", "coral_clinic_appt", "coral_prognosis"]
tasks = ["coral_clinic_appt", "coral_prognosis", "coral_regime_changes"]


# Function to process and save responses for a given task
def process_and_save_responses(task_name):
    for model in models:
        for type_ in types:
            response_file_path = os.path.join(
                data_dir, f"api_responses/{model}/{task_name}_{type_}_responses.jsonl"
            )
            responses = load_api_responses(response_file_path)

            # Bind responses to DataFrame
            df = bind_responses_to_df(responses, task_name, model, type_)

            # Save the DataFrame to CSV
            output_file_path = os.path.join(
                output_dir, f"{model}_{type_}_responses.csv"
            )
            df.to_csv(output_file_path, index=False)
            print(f"Saved responses for {model} {type_} to {output_file_path}")

        # Join the two types for each model
        df_brand = pd.read_csv(os.path.join(output_dir, f"{model}_brand_responses.csv"))
        df_generic = pd.read_csv(
            os.path.join(output_dir, f"{model}_generic_responses.csv")
        )

        print(f"Length of brand responses: {len(df_brand)}")
        print(f"Length of generic responses: {len(df_generic)}")

        model_df = pd.concat([df_brand, df_generic], axis=0)
        model_out_path = os.path.join(
            output_dir, f"{model}/{task_name}/all_responses.csv"
        )

        # Get the directory path
        model_out_dir = os.path.dirname(model_out_path)

        # Create the directory if it doesn't exist
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)

        # Write the DataFrame to a CSV file
        model_df.to_csv(model_out_path, index=False)

        # Delete the individual type files
        os.remove(os.path.join(output_dir, f"{model}_brand_responses.csv"))
        os.remove(os.path.join(output_dir, f"{model}_generic_responses.csv"))

        # Print shape, unique types, and head of the combined DataFrame
        print(f"Combined responses for {model}:")
        print(f"Shape: {model_df.shape}")
        print(f"Unique types: {model_df['type'].unique()}")
        print(model_df.head())
        print("\n")

        # Process the responses based on the task
        if "sentiment" in task_name:
            results_df = process_sentiment(model_df, model_out_dir, task_name, model)
        elif (
            "clinic_appt" in task_name
            or "prognosis" in task_name
            or "regime_changes" in task_name
        ):
            save_statistics(model_df, model, task_name, output_dir)


# Process each task
for task in tasks:
    process_and_save_responses(task)

print("Processing completed for all tasks.")
