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
    dataset_name,
):
    batch_tasks = []
    for temperature in temperatures:
        for _, row in data.iterrows():
            user_message_content = row["prompt"]
            unique_id = row["unique_id"]
            task_id = (
                f"{unique_id}_{task_name}_{dataset_name}_" f"{temperature}_{model_name}"
            )
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
    DEBUG = False

    irae_data_dir = "src/irAE"
    request_dir = "data/request/"

    # Read in IrAE CSV files
    irae_df_brand = pd.read_csv(
        os.path.join(irae_data_dir, "generated_prompts_brand_only.csv")
    )
    irae_df_generic = pd.read_csv(
        os.path.join(irae_data_dir, "generated_prompts_generic_only.csv")
    )

    if DEBUG:
        irae_df_brand = irae_df_brand.head(50)
        irae_df_generic = irae_df_generic.head(50)

    temperatures = [0.0, 0.7, 1.0]
    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"]

    system_prompts = {
        "differential": "You are an expert Oncologist reviewing a note from the previous visit. Your task is to review the note step by step and provide a Python list of diagnoses in order of probability.",
        "irae_detection": "You are an expert Oncologist reviewing a note from the previous visit. Your task is to review the note step by step and answer the question using only a single number.",
    }

    max_tokens = {"differential": 250, "irae_detection": 1}

    datasets = [
        ("brand_only", irae_df_brand),
        ("generic_only", irae_df_generic),
    ]

    # convert unique_id to snake case
    irae_df_brand["unique_id"] = irae_df_brand["unique_id"].apply(
        lambda x: x.replace(" ", "_")
    )
    irae_df_generic["unique_id"] = irae_df_generic["unique_id"].apply(
        lambda x: x.replace(" ", "_")
    )

    for dataset_name, dataset in datasets:
        for task_name in ["differential", "irae_detection"]:
            filtered_irae_df = dataset[dataset["task_name"] == task_name]
            for model in models:
                all_tasks = []

                batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
                    filtered_irae_df,
                    model_name=model,
                    temperatures=temperatures,
                    max_tokens=max_tokens[task_name],
                    system_prompt=system_prompts[task_name],
                    task_name=f"{task_name}",
                    dataset_name=f"{dataset_name}",
                )
                all_tasks.extend(batch_api_payload_jsonl)

                jsonl_file_path = os.path.join(
                    request_dir,
                    f"batch_{task_name}_{dataset_name}",
                    f"{model}_all_temperatures.jsonl",
                )
                if not os.path.exists(os.path.dirname(jsonl_file_path)):
                    os.makedirs(os.path.dirname(jsonl_file_path))

                with open(jsonl_file_path, "w") as file:
                    for line in all_tasks:
                        file.write(line + "\n")
                print(f"Saved {jsonl_file_path}")

    print("Processing complete.")
