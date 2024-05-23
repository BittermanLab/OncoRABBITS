import pandas as pd
import numpy as np
import os
import json


def load_combined_notes(data_dir):
    combined_notes = pd.read_csv(os.path.join(data_dir, "combined_notes.csv"))

    print("Original data shape:", combined_notes.shape)
    print("Columns:", combined_notes.columns)
    print("First row:")
    print(combined_notes.head(1))
    print(f"\n" * 5)

    return combined_notes


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

    data_dir = "src/coral_count/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/unannotated/data/"
    request_dir = "data/request/"

    combined_df = load_combined_notes(data_dir)

    if DEBUG:
        half_len = len(combined_df) // 2
        combined_df = combined_df.head(half_len)

    temperatures = [0.0, 0.7, 1.0]
    max_tokens = 1
    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
    system_prompt = "You are an expert Oncologist reviewing a report your resident has prepared. Read this step by step and give an estimate for the prognosis of this patient. Give your answer in months only e.g for 6 months write '6'."
    user_prompt_template = "Report: {note_text} \nQuestion: In months, what is the prognosis of this patient?:\nAnswer: "
    task_name = "coral_prognosis"
    all_tasks = []
    for model in models:
        batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
            combined_df,
            model_name=model,
            temperatures=temperatures,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            task_name=task_name,
        )
        all_tasks.extend(batch_api_payload_jsonl)

    jsonl_file_path = os.path.join(
        request_dir, "batch_coral_extract_all_models_all_temperatures.jsonl"
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
