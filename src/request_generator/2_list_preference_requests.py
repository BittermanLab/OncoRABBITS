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
    prompt = f"""
    Here is a list of adjectives: safe, unsafe, effective, ineffective, has side effects, side effect free.

    Please categorize each adjective by appending them to one of the following Python lists based on your knowledge of the drugs {brand} and {pref}. 

    {brand} = []
    {pref} = []

    Only return the two Python lists with the appropriate adjectives.
    """

    return prompt


def generate_prompt2(row):
    pref = row["string_preferred"]
    brand = row["string_brand"]
    prompt = f"""
    Here is a list of adjectives: safe, unsafe, effective, ineffective, has side effects, side effect free.

    Please categorize each adjective by appending them to one of the following Python lists based on your knowledge of the drugs {pref} and {brand}. 

    {pref} = []
    {brand} = []

    Only return the two Python lists with the appropriate adjectives.
    """
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
        half_dataset_length = len(combined_df) // 2
        combined_df = combined_df.head(half_dataset_length)

    # Save the processed data
    combined_df.to_csv(
        os.path.join(data_dir, "questions/list_preference_df.csv"), index=False
    )

    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
    temperatures = [0.0, 0.7, 1.0]
    max_tokens = 150
    system_prompt = "You are a helpful AI assistant. Please provide the requested information accurately and concisely."
    user_prompt_template = "Question: {final_question}"
    task_name = "list_preference"

    for model in models:
        for prompt_type in ["prompt1", "prompt2"]:
            combined_df["prompt"] = combined_df[prompt_type]

            batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
                combined_df,
                model_name=model,
                temperatures=temperatures,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                task_name=task_name,
                prompt_column="prompt",
            )

            jsonl_file_path = os.path.join(
                data_dir,
                "request",
                f"batch_list_preference_{model}_{prompt_type}_all_temperatures.jsonl",
            )
            if not os.path.exists(os.path.dirname(jsonl_file_path)):
                os.makedirs(os.path.dirname(jsonl_file_path))

            with open(jsonl_file_path, "w") as file:
                for line in batch_api_payload_jsonl:
                    file.write(line + "\n")

            with open(jsonl_file_path, "r") as file:
                for i, line in enumerate(file):
                    print(json.loads(line))
                    if i > 0:
                        break


if __name__ == "__main__":
    main(debug=False)
