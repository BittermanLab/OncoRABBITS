import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# root of project
root_dir = "../../"
# cache directory
cache_dir = root_dir + "../cache/"


def load_drug_map(csv_path, reverse_map=False):
    df = pd.read_csv(csv_path)
    return dict(
        zip(
            df["generic"] if reverse_map else df["brand"],
            df["brand"] if reverse_map else df["generic"],
        )
    )


def load_hf_dataset(dataset_name="augtoma/usmle_step_2", split="train", cache_dir=None):
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def format_question(instruction, question, options):
    formatted_instruction = f"Instruction: {instruction}"
    formatted_question = f"Question: {question}"
    formatted_options = (
        "\n".join([f"{k}: {v}" for k, v in options.items()]) if options else ""
    )
    formatted_question = f"{formatted_instruction}\n{formatted_question}\nOptions:\n{formatted_options}\nAnswer: "
    return formatted_question


def format_and_flag_questions(dataset, keywords):
    formatted_data = []
    for example in dataset:
        instruction = "Answer this multiple choice question with the corresponding letter of the answer choice only."
        question = example["question"]
        options = example.get("options", {})
        formatted_question = format_question(instruction, question, options)
        keyword_flags = [
            keyword for keyword in keywords if keyword.lower() in question.lower()
        ]

        formatted_data.append(
            {
                "formatted_question": formatted_question,
                "answer": example["answer"],
                "answer_idx": example["answer_idx"],
                "options": options,
                "contains_keyword": bool(keyword_flags),
                "found_keywords": ", ".join(keyword_flags),
            }
        )
    return pd.DataFrame(formatted_data)


def filter_questions_by_flag(dataframe):
    return dataframe[dataframe["contains_keyword"] == True]


def replace_keywords(dataframe, keyword_map, mode="brand_to_generic"):
    altered_data = []
    for index, row in dataframe.iterrows():
        altered_question = row["formatted_question"]
        for old_keyword, new_keyword in keyword_map.items():
            if mode == "brand_to_generic":
                altered_question = altered_question.replace(old_keyword, new_keyword)
            elif mode == "generic_to_brand":
                altered_question = altered_question.replace(new_keyword, old_keyword)
            elif mode == "swap_all":
                if old_keyword in altered_question:
                    altered_question = altered_question.replace(
                        old_keyword, new_keyword
                    )
                elif new_keyword in altered_question:
                    altered_question = altered_question.replace(
                        new_keyword, old_keyword
                    )

        row["formatted_question"] = altered_question
        altered_data.append(row)
    return pd.DataFrame(altered_data)


def generate_questions(dataset_name, split, csv_path, filter, mode):
    brand_to_generic = load_drug_map(csv_path)
    generic_to_brand = load_drug_map(csv_path, reverse_map=True)
    all_keywords = set(brand_to_generic.keys()).union(
        brand_to_generic.values(), generic_to_brand.keys(), generic_to_brand.values()
    )
    print(f"Total keywords: {len(all_keywords)}")

    dataset = load_hf_dataset(dataset_name, split)
    formatted_questions = format_and_flag_questions(dataset, all_keywords)
    print(f"Total questions: {len(formatted_questions)}")
    print(f"Questions with keywords: {formatted_questions['contains_keyword'].sum()}")

    if filter:
        filtered_questions = filter_questions_by_flag(formatted_questions)
        print(f"Filtered questions: {len(filtered_questions)}")
    else:
        filtered_questions = formatted_questions

    keyword_map = (
        brand_to_generic
        if mode == "brand_to_generic"
        else (
            generic_to_brand
            if mode == "generic_to_brand"
            else {**brand_to_generic, **{v: k for k, v in generic_to_brand.items()}}
        )
    )
    if mode != "NA":
        replaced_questions = replace_keywords(filtered_questions, keyword_map, mode)
        print(f"Replaced questions: {len(replaced_questions)}")
    else:
        replaced_questions = filtered_questions

    return replaced_questions


if __name__ == "__main__":
    questions_df = generate_questions(
        "augtoma/usmle_step_2",
        "train",
        "src/med_tasks/drug_names.csv",
        True,
        "swap_all",
    )
    print(questions_df.head())
