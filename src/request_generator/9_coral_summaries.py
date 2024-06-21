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

    coral_data_dir = "src/coral_count/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/unannotated/data"
    request_dir = "data/request/"

    # read in coral csv
    coral_df = pd.read_csv(os.path.join(coral_data_dir, "combined_notes.csv"))

    if DEBUG:
        coral_df = coral_df.head(5)

    temperatures = [0.0, 0.7, 1.0]
    max_tokens = 500
    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
    example_1 = "Summary: \n Age: 55 years \n Gender: Male \n Cancer diagnosis: Stage III non-small cell lung cancer (NSCLC) \n PMH: hypertension, hyperlipidemia \n Prior cancer treatments: None \n Current cancer treatments: radiotherapy with concurrent cisplatin (started 2 weeks ago) \n Current medication list: lisinopril, amlodipine, simvastatin, aspirin, pantoprazole \n Summary of most recent oncology visit (1 week ago): 55-year-old male with newly diagnosed stage III NSCLC.  \n He is on chemoradiation and tolerating treatment well. No significant side effects were reported. Will continue treatment as planned."
    example_2 = "Summary: \n Age: 47 years \n Gender: Female \n Cancer diagnosis: Stage II invasive ductal carcinoma of the breast \n PMH: asthma, obesity \n Prior cancer treatments: lumpectomy (completed 2 months ago) \n Current cancer treatments: adjuvant doxorubicin/cyclophosphamide (started 1 month ago) \n Current medication list: albuterol, montelukast, metformin, aspirin, atorvastatin, vitamin D \n Summary of most recent oncology visit (3 weeks ago): 47-year-old female with a history of stage II breast cancer s/p lumpectomy. She is on adjuvant doxorubicin/cyclophosphamide and tolerating treatment well. Will continue treatment as planned."

    system_prompt = "You are an expert Oncologist reviewing a note from the previous visit. Your task is to review the note step by step and summarize the note into the format of the examples below. \n\nExample 1: {example_1} \n\nExample 2: {example_2}".format(
        example_1=example_1, example_2=example_2
    )
    user_prompt_template = "Report to summarize: {note_text} \n Summary: \n"

    all_tasks = []

    # Brand
    task_name = "coral_summary"
    for model in models:
        print(f"Running {task_name} for {model}")
        all_tasks = []

        batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
            coral_df,
            model_name=model,
            temperatures=temperatures,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            task_name=task_name,
        )
        all_tasks.extend(batch_api_payload_jsonl)

        jsonl_file_path = os.path.join(
            request_dir, f"batch_{task_name}_{model}_all_temperatures.jsonl"
        )
        if not os.path.exists(os.path.dirname(jsonl_file_path)):
            os.makedirs(os.path.dirname(jsonl_file_path))

        with open(jsonl_file_path, "w") as file:
            for line in all_tasks:
                file.write(line + "\n")
        print(f"Saved {jsonl_file_path}")
