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


# def strip_special_chars(input_str):
#     """Remove special characters from string start/end"""
#     if not input_str:
#         return input_str

#     start_index = 0
#     end_index = len(input_str) - 1

#     while (
#         start_index < len(input_str)
#         and input_str[start_index] not in string.ascii_letters + string.digits
#     ):
#         start_index += 1

#     while (
#         end_index >= 0
#         and input_str[end_index] not in string.ascii_letters + string.digits
#     ):
#         end_index -= 1

#     if start_index <= end_index:
#         return input_str[start_index : end_index + 1]
#     else:
#         return ""


# def starts_with_capital_letter(input_str):
#     """Check if the response starts correctly with a capital letter and follows MCQ answer formats."""
#     pattern = r"^[A-Z](:|\.|) .+"
#     return bool(re.match(pattern, input_str))


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

    # Save results after processing all questions
    with open(outname, "w") as file:
        json.dump(answers, file)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process USMLE Dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="usmle_step_2",
        choices=["usmle_step_2", "mmlu_professional_medicine"],
        help="Hugging Face dataset to load (default: usmle_step_2)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/gpt-neo-2.7B",
        choices=[
            "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/pythia-70m",
            "BioMistral/BioMistral-7B",
        ],
        help="Model to use for evaluation (default: EleutherAI/gpt-neo-2.7B)",
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
    parser.add_argument(
        "--swap_mode",
        type=str,
        default="NA",
        choices=["brand_to_generic", "generic_to_brand", "swap_all", "NA"],
        help="Mode to swap keywords in questions (default: NA)",
    )

    args = parser.parse_args()

    ## Generate questions
    from drug_replace import generate_questions

    questions_df = generate_questions(
        args.dataset_name, args.split, args.csv_path, args.filter, args.swap_mode
    )

    print(questions_df.head())

    ## Models

    # create run directory
    model_dir = args.model_name.split("/")[-1]
    dataset_dir = args.dataset_name.split("/")[-1]

    output_dir = os.path.join(args.output_dir, dataset_dir, model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_path = os.path.join(output_dir, f"{args.filter}_{args.swap_mode}.json")

    # Load the model
    model, tokenizer = load_model(args.model_name)
    results = evaluate_multiple_choice_questions(
        model, tokenizer, questions_df, results_path
    )

    print(results.head())

    # Summarize results
    summary = summarize_results(results)
    print(summary)

    summary.to_csv(
        os.path.join(output_dir, f"{args.filter}_{args.swap_mode}_summary.csv"),
        index=False,
    )
