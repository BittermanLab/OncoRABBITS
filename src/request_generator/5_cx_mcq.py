import pandas as pd
import json
import uuid


def load_parquet_files():
    generic_df = pd.read_parquet(
        "data/questions/drugs_contraindications_generic.parquet"
    )
    brand_df = pd.read_parquet("data/questions/drugs_contraindications_brand.parquet")
    return generic_df, brand_df


def prepare_data_for_batch(df, type):

    # add a unique_id column which is index + brand or generic
    df["unique_id"] = df.index.astype(str) + "_" + type
    # Prepare the data in the format needed for the API request
    df["question"] = df["sent1"]
    df["choices"] = df.apply(
        lambda row: [row["ending0"], row["ending1"], row["ending2"], row["ending3"]],
        axis=1,
    )
    df["correct_choice"] = df["label"]

    return df[["unique_id", "question", "choices", "correct_choice"]]


def generate_batch_api_payload_jsonl(
    data,
    model_name,
    temperatures,
    max_tokens,
    system_prompt,
    user_prompt_template,
    task_name,
    type,
):
    batch_tasks = []
    for temperature in temperatures:
        for _, row in data.iterrows():
            user_message_content = user_prompt_template.format(**row)
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


def main():
    # Load the parquet files
    generic_df, brand_df = load_parquet_files()

    # Prepare data for batch processing
    generic_data = prepare_data_for_batch(generic_df, "generic")
    brand_data = prepare_data_for_batch(brand_df, "brand")

    print(generic_data.head())
    print(brand_data.head())

    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
    temperatures = [0.0, 0.7, 1.0]
    max_tokens = 1
    system_prompt = "You are a helpful AI assistant. Please provide the requested information accurately and concisely. You will be given multiple choice questions. You must only answer the questions with only the corresponding numbers for the answer."
    user_prompt_template = "Question: {question}\n\nChoices:\n1. {choices[0]}\n2. {choices[1]}\n3. {choices[2]}\n4. {choices[3]}\n\nPlease provide the correct answer (1, 2, 3, or 4) only. Do not provide any other information."

    for model_name in models:
        # Generate batch API payloads
        generic_tasks = generate_batch_api_payload_jsonl(
            generic_data,
            model_name,
            temperatures,
            max_tokens,
            system_prompt,
            user_prompt_template,
            "cx_mcq",
            "generic",
        )

        brand_tasks = generate_batch_api_payload_jsonl(
            brand_data,
            model_name,
            temperatures,
            max_tokens,
            system_prompt,
            user_prompt_template,
            "cx_mcq",
            "brand",
        )

        # Save the batch tasks to JSONL files
        with open(
            f"data/request/batch_cx_mcq_generic_only/{model_name}_all_temperatures.jsonl",
            "w",
        ) as f:
            for task in generic_tasks:
                f.write(f"{task}\n")

        with open(
            f"data/request/batch_cx_mcq_brand_only/{model_name}_all_temperatures.jsonl",
            "w",
        ) as f:
            for task in brand_tasks:
                f.write(f"{task}\n")

        print("Batch API payload JSONL files created successfully.")


if __name__ == "__main__":
    main()
