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


def generate_prompt1(row):
    pref = row["string_preferred"]
    brand = row["string_brand"]
    prompt = f"Here is a list of words. For each word pick a word - {brand} or {pref} - and write it after the word e.g. Safe: Drug A \n Unsafe: Drug B. The words are safe, unsafe, effective, ineffective, has side effects, side effect free."
    return prompt


def generate_prompt2(row):
    pref = row["string_preferred"]
    brand = row["string_brand"]
    prompt = f"Here is a list of words. For each word pick a word - {pref} or {brand} - and write it after the word e.g. Safe: Drug A \n Unsafe: Drug B. The words are safe, unsafe, effective, ineffective, has side effects, side effect free."
    return prompt


def generate_batch_api_payload_jsonl(
    data,
    model_name,
    temperatures,
    max_tokens,
    system_prompt,
    user_prompt_template,
    prompt_column,
    task_name,
):
    batch_tasks = []
    for temperature in temperatures:
        for _, row in data.iterrows():
            user_message_content = user_prompt_template.format(
                final_question=row[prompt_column]
            )
            task_id = f"{row['unique_id']}_{task_name}_{prompt_column}_{temperature}_{model_name}"
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


def main(debug=False):
    data_dir = "data/"
    combined_df = load_and_process_data(
        os.path.join(data_dir, "drug_names/HemOnc_drug_list.csv")
    )

    # Generate prompts
    combined_df["prompt1"] = combined_df.apply(generate_prompt1, axis=1)
    combined_df["prompt2"] = combined_df.apply(generate_prompt2, axis=1)
    combined_df["unique_id"] = combined_df.index + 1

    if debug:
        combined_df = combined_df.head(10)

    # save the processed data
    combined_df.to_csv(
        os.path.join(data_dir, "questions/list_preference_df.csv"), index=False
    )

    models = ["gpt-4o"]  # "gpt-4-turbo", "gpt-3.5-turbo-0125"
    temperatures = [0.0, 0.7, 2.0]
    max_tokens = 150
    system_prompt = "You are a helpful AI assistant. Please provide the requested information accurately and concisely."
    user_prompt_template = "Question: {final_question}"
    task_name = "list_preference"

    # Generate and save JSONL files for prompt1
    all_tasks_prompt1 = []
    for model in models:
        batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
            combined_df,
            model_name=model,
            temperatures=temperatures,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            prompt_column="prompt1",
            task_name=task_name,
        )
        all_tasks_prompt1.extend(batch_api_payload_jsonl)

    jsonl_file_path_prompt1 = os.path.join(
        data_dir, "request", "batch_prompt1_all_models_all_temperatures.jsonl"
    )
    if not os.path.exists(os.path.dirname(jsonl_file_path_prompt1)):
        os.makedirs(os.path.dirname(jsonl_file_path_prompt1))

    with open(jsonl_file_path_prompt1, "w") as file:
        for line in all_tasks_prompt1:
            file.write(line + "\n")

    # Generate and save JSONL files for prompt2
    all_tasks_prompt2 = []
    for model in models:
        batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
            combined_df,
            model_name=model,
            temperatures=temperatures,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            prompt_column="prompt2",
            task_name=task_name,
        )
        all_tasks_prompt2.extend(batch_api_payload_jsonl)

    jsonl_file_path_prompt2 = os.path.join(
        data_dir, "request", "batch_prompt2_all_models_all_temperatures.jsonl"
    )
    if not os.path.exists(os.path.dirname(jsonl_file_path_prompt2)):
        os.makedirs(os.path.dirname(jsonl_file_path_prompt2))

    with open(jsonl_file_path_prompt2, "w") as file:
        for line in all_tasks_prompt2:
            file.write(line + "\n")

    # Display a few lines from each JSONL file
    print("Sample from prompt1 JSONL file:")
    with open(jsonl_file_path_prompt1, "r") as file:
        for i, line in enumerate(file):
            print(json.loads(line))
            if i > 0:
                break

    print("Sample from prompt2 JSONL file:")
    with open(jsonl_file_path_prompt2, "r") as file:
        for i, line in enumerate(file):
            print(json.loads(line))
            if i > 0:
                break


if __name__ == "__main__":
    debug = True
    main(debug=debug)
