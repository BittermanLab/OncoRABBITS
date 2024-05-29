import pandas as pd
import numpy as np
import os
import json


def load_and_process_data(file_path):
    df = pd.read_csv(file_path)

    # Filter out rows that are not "preferred name" or "brand name"
    filtered_df = df[df["string_type"].isin(["preferred name", "brand name"])]

    # Group by concept_code and keep the first occurrence of each string_type per group
    unique_names_df = (
        filtered_df.groupby(["concept_code", "string_type"]).first().reset_index()
    )

    # Since we want to keep one of each type per concept_code, let's ensure there's only one of each
    final_df = unique_names_df.groupby("concept_code").filter(lambda x: len(x) <= 2)

    # Filter out concept_codes that appear only once in the DataFrame
    counts = final_df["concept_code"].value_counts()
    filtered_final_df = final_df[
        final_df["concept_code"].isin(counts[counts > 1].index)
    ]

    # Ensure we have a DataFrame that includes both a preferred name and a brand name for each concept_code
    concept_codes_with_both_names = filtered_final_df.groupby("concept_code").filter(
        lambda x: len(x) == 2
    )

    # Split the DataFrame into two: one for preferred names and one for brand names
    preferred_names_df = concept_codes_with_both_names[
        concept_codes_with_both_names["string_type"] == "preferred name"
    ]

    brand_names_df = concept_codes_with_both_names[
        concept_codes_with_both_names["string_type"] == "brand name"
    ]

    # Merge them to have a single DataFrame with both preferred and brand names for each concept_code
    combined_df = pd.merge(
        preferred_names_df,
        brand_names_df,
        on="concept_code",
        suffixes=("_preferred", "_brand"),
    )

    return combined_df


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
    system_prompt = "You are an expert Oncologist reviewing a report your resident has prepared. Read this step by step and determine whether any changes to the cancer medication plan are needed. Answer Yes or No only."
    user_prompt_template = "Report: {note_text} \nQuestion: Yes or No, are changes to the cancer medication plan needed?:\nAnswer: "

    all_tasks = []

    # Brand
    task_name = "coral_regime_change_brand"
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
    task_name = "coral_regime_change_generic"
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
        request_dir, "batch_coral_regime_change_all_models_all_temperatures.jsonl"
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
