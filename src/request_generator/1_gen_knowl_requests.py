import pandas as pd
import numpy as np
import os
import json


def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    filtered_df = df[df["string_type"].isin(["preferred name", "brand name"])]
    unique_names_df = (
        filtered_df.groupby(["concept_code", "string_type"]).first().reset_index()
    )
    final_df = unique_names_df.groupby("concept_code").filter(lambda x: len(x) <= 2)
    counts = final_df["concept_code"].value_counts()
    filtered_final_df = final_df[
        final_df["concept_code"].isin(counts[counts > 1].index)
    ]
    concept_codes_with_both_names = filtered_final_df.groupby("concept_code").filter(
        lambda x: len(x) == 2
    )
    return concept_codes_with_both_names


def get_incorrect_answers(correct_answer, all_answers):
    incorrect_answers = np.random.choice(
        all_answers[all_answers != correct_answer], size=3, replace=False
    )
    return incorrect_answers.tolist()


def generate_variations(
    all_answers,
    correct_answer,
    question_template,
    concept_code,
    variation_id,
    num_variations=5,
):
    incorrect_answers = get_incorrect_answers(correct_answer, all_answers)
    options = [correct_answer] + incorrect_answers
    np.random.shuffle(options)
    return {
        "concept_code": concept_code,
        "variation_id": variation_id,
        "question": question_template,
        "options": options,
        "correct_answer": correct_answer,
    }


def generate_questions(
    combined_df, all_preferred_names, all_brand_names, num_variations=5
):
    questions_answers = []
    for index, row in combined_df.iterrows():
        concept_code = row["concept_code"]
        preferred_name = row["string_preferred"]
        brand_name = row["string_brand"]
        for variation_id in range(1, num_variations + 1):
            question_brand_template = f"Which of the following is the brand name for the drug {preferred_name}?"
            variation_brand = generate_variations(
                all_brand_names,
                brand_name,
                question_brand_template,
                concept_code,
                variation_id,
                num_variations,
            )
            questions_answers.append(variation_brand)
            question_generic_template = (
                f"Which of the following is the generic name for the drug {brand_name}?"
            )
            variation_generic = generate_variations(
                all_preferred_names,
                preferred_name,
                question_generic_template,
                concept_code,
                variation_id,
            )
            questions_answers.append(variation_generic)
    return questions_answers


def expand_combined_df(combined_df, num_variations):
    expanded_combined_df = pd.DataFrame(
        np.repeat(combined_df.values, num_variations, axis=0),
        columns=combined_df.columns,
    )
    expanded_combined_df["variation_id"] = np.tile(
        np.arange(1, num_variations + 1), len(combined_df)
    )
    return expanded_combined_df


def generate_name_questions(row):
    questions = {}
    for type in ["preferred", "brand"]:
        question_type_key = f"question_{type}"
        options_type_key = f"options_{type}"
        question = row[question_type_key]
        options = row[options_type_key]
        if not isinstance(options, list):
            options_list = options.split(",") if isinstance(options, str) else []
        else:
            options_list = options
        options_formatted = ", ".join(
            [f"{i + 1}: {opt.strip()}" for i, opt in enumerate(options_list)]
        )
        final_question = f"{question}\nOptions: {options_formatted}\nAnswer: "
        questions[type] = final_question
    return questions["preferred"], questions["brand"]


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


if __name__ == "__main__":
    DEBUG = False

    data_dir = "../../data/"
    combined_df = pd.read_csv(os.path.join(data_dir, "drug_names/combined_df.csv"))

    all_preferred_names = combined_df["string_preferred"].unique()
    all_brand_names = combined_df["string_brand"].unique()

    questions_answers = generate_questions(
        combined_df, all_preferred_names, all_brand_names, num_variations=5
    )
    qa_df = pd.DataFrame(questions_answers)
    qa_brand_df = qa_df[
        qa_df["question"].str.contains("brand name for the drug")
    ].reset_index(drop=True)
    qa_preferred_df = qa_df[
        qa_df["question"].str.contains("generic name for the drug")
    ].reset_index(drop=True)

    expanded_combined_df = expand_combined_df(combined_df, num_variations=5)
    filtered_final_df_with_qa = pd.merge(
        expanded_combined_df,
        qa_brand_df,
        on=["concept_code", "variation_id"],
        how="left",
        suffixes=("_brand", ""),
    )
    filtered_final_df_with_qa = pd.merge(
        filtered_final_df_with_qa,
        qa_preferred_df,
        on=["concept_code", "variation_id"],
        how="left",
        suffixes=("_brand", "_preferred"),
    )

    filtered_final_df_with_qa[["final_preferred_question", "final_brand_question"]] = (
        filtered_final_df_with_qa.apply(
            lambda row: generate_name_questions(row), axis=1, result_type="expand"
        )
    )

    brand_questions_df = filtered_final_df_with_qa[
        [
            "concept_code",
            "string_type_brand",
            "string_brand",
            "question_brand",
            "options_brand",
            "correct_answer_brand",
            "final_brand_question",
        ]
    ].copy()
    brand_questions_df["type"] = "brand"
    brand_questions_df.rename(
        columns={
            "string_type_brand": "string_type",
            "string_brand": "string",
            "question_brand": "question",
            "options_brand": "options",
            "correct_answer_brand": "correct_answer",
            "final_brand_question": "final_question",
        },
        inplace=True,
    )

    preferred_questions_df = filtered_final_df_with_qa[
        [
            "concept_code",
            "string_type_preferred",
            "string_preferred",
            "question_preferred",
            "options_preferred",
            "correct_answer_preferred",
            "final_preferred_question",
        ]
    ].copy()
    preferred_questions_df["type"] = "preferred"
    preferred_questions_df.rename(
        columns={
            "string_type_preferred": "string_type",
            "string_preferred": "string",
            "question_preferred": "question",
            "options_preferred": "options",
            "correct_answer_preferred": "correct_answer",
            "final_preferred_question": "final_question",
        },
        inplace=True,
    )

    # Concatenate the DataFrames
    combined_questions_df = pd.concat([brand_questions_df, preferred_questions_df])
    combined_questions_df.reset_index(drop=True, inplace=True)
    combined_questions_df["unique_id"] = combined_questions_df.index + 1

    if DEBUG:
        half_dataset_length = len(combined_questions_df) // 2
        combined_questions_df = combined_questions_df.tail(half_dataset_length)

    # save gen_knowl_df
    combined_questions_df.to_csv(
        os.path.join(data_dir, "questions/general_knowledge_df.csv"),
        index=False,
    )

    # print unique string types
    print(combined_questions_df["string_type"].unique())
    print(combined_questions_df.head())
    print(combined_questions_df.tail())
    print(combined_questions_df.shape)

    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
    temperatures = [0.0, 0.7, 1.0]
    max_tokens = 1
    system_prompt = "You are a helpful AI assistant. Please provide the requested information accurately and concisely. You will be given multiple choice questions. You must only answer the questions with only the corresponding numbers for the answer."
    user_prompt_template = "Question: {final_question}"
    task_name = "general_knowledge"

    for model in models:
        batch_api_payload_jsonl = generate_batch_api_payload_jsonl(
            combined_questions_df,
            model_name=model,
            temperatures=temperatures,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            task_name=task_name,
        )

        jsonl_file_path = os.path.join(
            data_dir,
            "request",
            f"batch_general_knowledge_{model}_all_temperatures.jsonl",
        )
        print(jsonl_file_path)
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
