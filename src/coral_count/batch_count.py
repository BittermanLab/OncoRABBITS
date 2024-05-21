import pandas as pd
import numpy as np
import os
import json


def load_and_process_data(data_dir):
    breast_df = pd.read_csv(os.path.join(data_dir, "breastca_unannotated.csv"))
    pancreatic_df = pd.read_csv(os.path.join(data_dir, "pdac_unannotated.csv"))

    breast_df["type"] = "breast"
    pancreatic_df["type"] = "pdac"

    combined_questions_df = pd.concat([breast_df, pancreatic_df], ignore_index=True)

    # create new unique index
    combined_questions_df["coral_idx"] = np.arange(len(combined_questions_df))

    print("Original data shape:", combined_questions_df.shape)
    print("Columns:", combined_questions_df.columns)
    print("First row:")
    print(combined_questions_df.head(1))
    print(f"\n" * 5)

    # save to csv
    combined_questions_df.to_csv(
        os.path.join(data_dir, "combined_notes.csv"), index=False
    )

    return combined_questions_df


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
    DEBUG = True

    data_dir = "src/coral_count/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/unannotated/data/"

    combined_df = load_and_process_data(data_dir)

    if DEBUG:
        combined_df = combined_df.head(10)

    models = ["gpt-4o"]  # "gpt-4-turbo", "gpt-3.5-turbo-0125"
    temperatures = [0.0, 0.7, 2.0]
    max_tokens = 250
    system_prompt = "You are a helpful AI assistant. Please provide the requested information accurately and concisely. You will be given a section of a medical report and must return a list of any clinical medications mentioned in the text."
    user_prompt_template = "Read the report below step by step and return a python list of any medications contained within: {note_text}"
    task_name = "coral_drug_extraction"

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
        data_dir, "request", "batch_coral_extract_all_models_all_temperatures.jsonl"
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
