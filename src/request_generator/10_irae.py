import pandas as pd
import numpy as np
import os
import json


def generate_batch_api_payload_jsonl(
    data,
    model_name,
    temperatures,
    max_tokens,
    system_prompt,
    task_name,
):
    batch_tasks = []
    for temperature in temperatures:
        for _, row in data.iterrows():
            user_message_content = row["prompt"]  # Access the prompt for each row
            task_id = f"{row['unique_id']}_{task_name}_{temperature}_{model_name}"
            task = {
                "custom_id": task_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message_content},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                },
            }
            batch_tasks.append(json.dumps(task))
    return batch_tasks


if __name__ == "__main__":
    DEBUG = True

    irae_data_dir = "src/irAE"
    request_dir = "data/request/"

    # read in irae csv
    irae_df = pd.read_csv(os.path.join(irae_data_dir, "generated_prompts.csv"))

    if DEBUG:
        irae_df = irae_df.head(5)

    temperatures = [0.0, 0.7, 1.0]
    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"]

    system_prompts = {
        "differential": "You are an expert Oncologist reviewing a note from the previous visit. Your task is to review the note step by step and provide a Python list of diagnoses in order of probability.",
        "irae_detection": "You are an expert Oncologist reviewing a note from the previous visit. Your task is to review the note step by step and answer the question using only a single number.",
    }

    max_tokens = {"differential": 50, "irae_detection": 1}

    for task_name in ["differential", "irae_detection"]:
        filtered_irae_df = irae_df[irae_df["task_name"] == task_name]
        for model in models:
            print(f"Running {task_name} for {model}")
            all_tasks = []

            batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
                filtered_irae_df,
                model_name=model,
                temperatures=temperatures,
                max_tokens=max_tokens[task_name],
                system_prompt=system_prompts[task_name],
                task_name=task_name,
            )
            all_tasks.extend(batch_api_payload_jsonl)

            jsonl_file_path = os.path.join(
                request_dir, f"batch_{task_name}", f"{model}_all_temperatures.jsonl"
            )
            if not os.path.exists(os.path.dirname(jsonl_file_path)):
                os.makedirs(os.path.dirname(jsonl_file_path))

            with open(jsonl_file_path, "w") as file:
                for line in all_tasks:
                    file.write(line + "\n")
            print(f"Saved {jsonl_file_path}")
