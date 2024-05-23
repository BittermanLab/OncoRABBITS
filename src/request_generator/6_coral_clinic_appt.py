import pandas as pd
import numpy as np
import os
import json


def load_data(data_dir):
    brand_df = pd.read_parquet(
        os.path.join(data_dir, "all_brand_to_generic_filtered/all.parquet")
    )
    generic_df = pd.read_parquet(
        os.path.join(data_dir, "all_generic_to_brand_filtered/all.parquet")
    )
    print("Brand data shape:", brand_df.shape)
    print("Columns:", brand_df.columns)

    print("Generic data shape:", generic_df.shape)
    print("Columns:", generic_df.columns)

    return brand_df, generic_df


def generate_batch_api_payload_jsonl(
    data,
    model_name,
    temperatures,
    max_tokens,
    system_prompt,
    user_prompt_template,
    task_name,
):
    batch_tasks = []
    for temperature in temperatures:
        for _, row in data.iterrows():
            user_message_content = user_prompt_template.format(**row)
            task_id = f"{row['coral_idx']}_{task_name}_{temperature}_{model_name}"
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

    data_dir = "data/coral/"
    request_dir = "data/request/"

    brand_df, generic_df = load_data(data_dir)
    if DEBUG:
        half_len = len(brand_df) // 2
        brand_df = brand_df.head(half_len)
        half_len = len(generic_df) // 2
        generic_df = generic_df.head(half_len)

    temperatures = [0.0, 0.7, 1.0]
    max_tokens = 1
    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
    system_prompt = "You are an expert Oncologist reviewing a report your resident has prepared. Read this step by step and determine when the next clinic appointment should be booked for. Give your answer in months only e.g for 6 months write '6'."
    user_prompt_template = "Report: {note_text} \nQuestion: In months, how many months should the next appointment be booked for in months?:\nAnswer: "
    task_name = "coral_clinic_appt"

    all_tasks = []
    # Brand
    task_name = "coral_clinic_appt_brand"
    for model in models:
        batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
            brand_df,
            model_name=model,
            temperatures=temperatures,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            task_name=task_name,
        )
        all_tasks.extend(batch_api_payload_jsonl)

    # Generic
    task_name = "coral_clinic_appt_generic"
    for model in models:
        batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
            generic_df,
            model_name=model,
            temperatures=temperatures,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            task_name=task_name,
        )
        all_tasks.extend(batch_api_payload_jsonl)

    jsonl_file_path = os.path.join(
        request_dir, "batch_coral_clinic_appt_all_models_all_temperatures.jsonl"
    )
    if not os.path.exists(os.path.dirname(jsonl_file_path)):
        os.makedirs(os.path.dirname(jsonl_file_path))

    with open(jsonl_file_path, "w") as file:
        for line in all_tasks:
            file.write(line + "\n")

    with open(jsonl_file_path, "r") as file:
        for i, line in enumerate(file):
            print(json.loads(line))
            if i > 0:
                break
