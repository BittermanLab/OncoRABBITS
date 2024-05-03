import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# root of project
root_dir = "../../"
# cache directory
cache_dir = root_dir + "../cache/"


def load_drug_map(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["old"], df["new"]))


def load_usmle_dataset(
    dataset_name="augtoma/usmle_step_2", split="train", cache_dir=cache_dir
):
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def format_question(question, options):
    formatted_question = question
    if options:
        formatted_options = "\n".join([f"{k}: {v}" for k, v in options.items()])
        formatted_question += "\nOptions:\n" + formatted_options
    return formatted_question


def format_and_flag_questions(dataset, keywords):
    formatted_data = []
    for example in dataset:
        question = example["question"]
        options = example.get("options", {})
        formatted_question = format_question(question, options)
        keyword_flags = [
            keyword for keyword in keywords if keyword.lower() in question.lower()
        ]

        formatted_data.append(
            {
                "formatted_question": formatted_question,
                "answer": example["answer"],
                "options": options,
                "contains_keyword": bool(keyword_flags),
                "found_keywords": ", ".join(keyword_flags),
            }
        )
    return pd.DataFrame(formatted_data)


def filter_questions_by_flag(dataframe):
    return dataframe[dataframe["contains_keyword"] == True]


def replace_keywords(dataframe, keyword_map):
    altered_data = []
    for index, row in dataframe.iterrows():
        altered_question = row["formatted_question"]
        for old_keyword, new_keyword in keyword_map.items():
            altered_question = altered_question.replace(old_keyword, new_keyword)
        altered_data.append(
            {
                "formatted_question": altered_question,
                "answer": row["answer"],
                "options": row["options"],
                "contains_keyword": row["contains_keyword"],
                "found_keywords": row["found_keywords"],
            }
        )
    return pd.DataFrame(altered_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process USMLE Dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="augtoma/usmle_step_2",
        help="Hugging Face dataset to load (default: augtoma/usmle_step_2)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--keywords", nargs="+", help="List of keywords to filter and flag in questions"
    )
    parser.add_argument(
        "--replace_map",
        nargs="+",
        help="Keyword replacement map, format: old:new",
        default=[],
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="src/med_tasks/drug_names.csv",
        help="Path to CSV file containing keywords",
    )

    args = parser.parse_args()

    # Parse keyword replacement map
    keyword_map = load_drug_map(args.csv_path)

    dataset = load_usmle_dataset(args.dataset_name, args.split)
    formatted_questions = format_and_flag_questions(dataset, keyword_map.keys())
    filtered_questions = filter_questions_by_flag(formatted_questions)
    replaced_questions = replace_keywords(filtered_questions, keyword_map)

    print("Filtered and replaced questions:")
    print(replaced_questions.head())
