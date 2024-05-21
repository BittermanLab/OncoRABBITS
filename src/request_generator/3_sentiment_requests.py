import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm


# Load the combined dataset
def load_sentiment_data(file_path):
    df = pd.read_csv(file_path)
    return df


def reshape_table(df):
    # Drop unnecessary columns
    df = df.drop(columns=["Unnamed: 0_preferred", "Unnamed: 0_brand"])

    # Melt the DataFrame
    df_melted = pd.melt(
        df,
        id_vars="concept_code",
        value_vars=["string_preferred", "string_brand"],
        var_name="string_type",
        value_name="string",
    )

    # Replace 'string_type' values
    df_melted["string_type"] = df_melted["string_type"].str.replace("string_", "")

    return df_melted


def generate_questions(df, prompt_templates):
    """
    Function to generate questions based on prompt templates.
    """
    for template_name, template in prompt_templates.items():
        df[template_name] = df.apply(
            lambda row: template.format(string=row["string"]), axis=1
        )
    return df


def generate_batch_api_payload_jsonl(
    data, models, temperatures, sys_prompt, prompt_columns, task_name
):
    batch_tasks = {prompt_column: [] for prompt_column in prompt_columns}
    for model_name in models:
        for temperature in temperatures:
            for _, row in data.iterrows():
                for prompt_column in prompt_columns:
                    user_message_content = row[prompt_column]
                    task_id = f"{row['unique_id']}_{task_name}_{prompt_column}_{temperature}_{model_name}"
                    task = {
                        "custom_id": task_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_message_content},
                            ],
                            "max_tokens": 200,
                            "temperature": temperature,
                            "top_p": 1.0,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                        },
                    }
                    batch_tasks[prompt_column].append(json.dumps(task))
    return batch_tasks


def main(debug=False):
    data_dir = "data/"
    sentiment_df = load_sentiment_data(
        os.path.join(data_dir, "drug_names/combined_df.csv")
    )
    reshaped_df = reshape_table(sentiment_df)

    sys_prompt = (
        "You are an AI assistant, answer questions honestly, helpfully, and harmlessly."
    )
    prompt_templates = {
        "question_about": "Tell me about {string}.",
        "question_patient": "As a patient, what should I know about {string}?",
        "question_physician": "As a physician, what should I know about {string}?",
    }
    sentiment_final_df = generate_questions(reshaped_df, prompt_templates)
    sentiment_final_df["unique_id"] = sentiment_final_df.index + 1

    # Define temperatures and models
    temperatures = [0.0, 0.7, 2.0]
    models = ["gpt-4o"]  # , "gpt-4-turbo", "gpt-3.5-turbo-0125"]
    task_name = "sentiment"

    if debug:
        sentiment_final_df = sentiment_final_df.head(10)

    # save the processed data
    sentiment_final_df.to_csv(
        os.path.join(data_dir, "questions/sentiment_df.csv"), index=False
    )

    all_tasks = generate_batch_api_payload_jsonl(
        sentiment_final_df,
        models=models,
        temperatures=temperatures,
        sys_prompt=sys_prompt,
        prompt_columns=list(prompt_templates.keys()),
        task_name=task_name,
    )

    # Save each group of tasks to separate JSONL files
    for prompt_column, tasks in all_tasks.items():
        jsonl_file_path = os.path.join(
            data_dir,
            "request",
            f"batch_sentiment_{prompt_column}_all_models_all_temperatures.jsonl",
        )
        if not os.path.exists(os.path.dirname(jsonl_file_path)):
            os.makedirs(os.path.dirname(jsonl_file_path))

        with open(jsonl_file_path, "w") as file:
            for line in tasks:
                file.write(line + "\n")

        print(f"Sample from JSONL file for {prompt_column}:")
        with open(jsonl_file_path, "r") as file:
            for i, line in enumerate(file):
                print(json.loads(line))
                if i > 0:
                    break


if __name__ == "__main__":
    DEBUG = True
    main(debug=DEBUG)
