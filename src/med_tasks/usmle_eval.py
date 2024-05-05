import json
import os
import re
import string
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from tqdm.auto import tqdm
import argparse

## Generate questions
from drug_replace import generate_questions

# root of project
root_dir = "../../"
# cache directory
cache_dir = root_dir + "../cache/"


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    model.eval()

    # set pad token id to eos token id if none
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def evaluate_multiple_choice_questions(model, tokenizer, questions_df, outname):
    """Evaluate a dataset of multiple-choice questions using a provided model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    answers = []

    for index, row in tqdm(questions_df.iterrows(), total=questions_df.shape[0]):
        formatted_question = row["formatted_question"]
        options = row["options"]
        correct_answer = row["answer"]
        answer_idx = row["answer_idx"]
        keyword_flags = row["contains_keyword"]
        keyword_list = row["found_keywords"]

        inputs = tokenizer(formatted_question, return_tensors="pt").to(device)

        # Prepare generation arguments to control the generation and score retrieval
        generate_args = {
            "max_length": inputs.input_ids.size(1)
            + 1,  # only generate one additional token
            "output_scores": True,
            "return_dict_in_generate": True,
            "do_sample": False,  # Avoid random sampling to get the most likely next token
        }

        generate_output = model.generate(**inputs, **generate_args)
        scores = generate_output.scores[0]  # scores for the first generated token
        softmax = torch.nn.functional.softmax(scores, dim=-1)

        # Get token IDs and corresponding log probabilities for expected answer labels (A to G)
        token_ids = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D", "E", "F", "G"])
        log_probs = softmax[:, token_ids]
        best_option_idx = torch.argmax(log_probs)
        best_option_token = tokenizer.convert_ids_to_tokens(token_ids[best_option_idx])

        # Save the result
        answers.append(
            {
                "question": formatted_question,
                "options": options,
                "actual_answer": correct_answer,
                "answer_idx": answer_idx,
                "predicted_answer": best_option_token,
                "correct": best_option_token == answer_idx,
                "keyword_flags": keyword_flags,
                "found_keywords": keyword_list,
            }
        )

    return pd.DataFrame(answers)


def summarize_results(results_df):
    total_questions = len(results_df)
    num_correct = results_df["correct"].sum()
    num_incorrect = total_questions - num_correct
    accuracy = (num_correct / total_questions) * 100 if total_questions else 0

    # Create summary data for the DataFrame
    data = {
        "Category": ["Overall"],
        "Total Questions": [total_questions],
        "Number Correct": [num_correct],
        "Number Incorrect": [num_incorrect],
        "Accuracy (%)": [round(accuracy, 2)],
    }

    # Adding keyword presence details
    keyword_presence = results_df["keyword_flags"].unique()
    for flag in [True, False]:  # Ensure handling for both conditions
        if flag in keyword_presence:
            subset = results_df[results_df["keyword_flags"] == flag]
            correct_count = subset["correct"].sum()
            question_count = len(subset)
            accuracy = (correct_count / question_count) * 100 if question_count else 0
            data["Category"].append(f"Keyword Presence {flag}")
            data["Total Questions"].append(question_count)
            data["Number Correct"].append(correct_count)
            data["Number Incorrect"].append(question_count - correct_count)
            data["Accuracy (%)"].append(round(accuracy, 2))
        else:
            data["Category"].append(f"Keyword Presence {flag}")
            data["Total Questions"].append(0)
            data["Number Correct"].append(0)
            data["Number Incorrect"].append(0)
            data["Accuracy (%)"].append(0)

    return pd.DataFrame(data)


def process_dataset(
    model_name, dataset_name, split, csv_path, output_dir, filter_questions, swap_mode
):
    print(
        f"Processing with dataset: {dataset_name}, model: {model_name}, swap mode: {swap_mode}, filter: {filter_questions}"
    )

    # Setup output directory
    model_dir = model_name.split("/")[-1]
    dataset_dir = dataset_name.split("/")[-1]
    results_folder = os.path.join(
        output_dir, dataset_dir, model_dir, f"{filter_questions}_{swap_mode}"
    )
    os.makedirs(results_folder, exist_ok=True)

    # Generate questions
    questions_df = generate_questions(
        dataset_name, split, csv_path, filter_questions, swap_mode
    )
    print(questions_df.head())

    # Load the model
    model, tokenizer = load_model(model_name)

    results = evaluate_multiple_choice_questions(
        model, tokenizer, questions_df, results_folder
    )

    # Save results
    results.to_json(os.path.join(results_folder, "results.json"))

    # Summarize and save results
    summary = summarize_results(results)
    summary.to_csv(os.path.join(results_folder, "summary.csv"), index=False)

    return summary


def main(args):
    models = [
        "EleutherAI/gpt-neo-2.7B",
        "EleutherAI/pythia-70m",
    ]
    datasets = ["usmle_step_2"]
    swap_modes = ["brand_to_generic", "generic_to_brand", "swap_all", "NA"]

    for dataset_name in datasets:
        for model_name in models:
            for swap_mode in swap_modes:
                process_dataset(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    split=args.split,
                    csv_path=args.csv_path,
                    output_dir=args.output_dir,
                    filter_questions=args.filter,
                    swap_mode=swap_mode,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process multiple datasets with different models and keyword replacement strategies"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="src/med_tasks/drug_names.csv",
        help="Path to CSV file containing keywords",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/med_tasks",
        help="Directory to save the output results.",
    )
    parser.add_argument(
        "--filter",
        type=bool,
        default=False,
        help="Only evaluate questions with flagged keywords.",
    )

    args = parser.parse_args()
    main(args)
