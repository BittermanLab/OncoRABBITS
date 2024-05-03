import json
import os
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm.auto import tqdm

# root of project
root_dir = "../../"
# cache directory
cache_dir = root_dir + "../cache/"


def load_model(model_name="EleutherAI/pythia-70m"):
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # set pad token id to eos token id if none
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def evaluate_multiple_choice_questions(
    model, tokenizer, questions_df, outname, ntries=5
):
    """Evaluate a dataset of multiple-choice questions using a provided model."""
    if os.path.exists(outname):
        with open(outname, "r") as file:
            answers = json.load(file)
    else:
        answers = []

    # Initialize the generation pipeline with explicit truncation
    nlp = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=0, truncation=True
    )

    pbar = tqdm(questions_df.iterrows(), total=questions_df.shape[0])
    for i, (index, question) in enumerate(pbar):
        if len(answers) > i:
            continue
        for j in range(ntries):
            # Generate answer with a clear distinction between input length and additional tokens to generate
            max_input_length = tokenizer.model_max_length
            prompt_length = len(tokenizer.encode(question["formatted_question"]))
            max_new_tokens = max(
                250, max_input_length - prompt_length
            )  # At least 50 tokens, adjust as needed

            try:
                generated_answers = nlp(
                    question["formatted_question"],
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                )
                response = generated_answers[0]["generated_text"].strip()
            except Exception as e:
                print(f"Error during generation: {e}")
                continue

            if is_valid_answer(response):
                break
            pbar.set_postfix_str(f"Retry {j+1}/{ntries}")
        answers.append(
            {
                "question": question["formatted_question"],
                "generated_answer": response,
                "actual_answer": question["answer"],
            }
        )
        with open(outname, "w") as file:
            json.dump(answers, file)

    return answers


def is_valid_answer(answer):
    """Check if the answer starts with a valid multiple choice prefix."""
    return answer.startswith(("A: ", "B: ", "C: ", "D: "))


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
        "--csv_path",
        type=str,
        default="src/med_tasks/drug_names.csv",
        help="Path to CSV file containing keywords",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the output results.",
    )
    parser.add_argument(
        "--ntries",
        type=int,
        default=5,
        help="Number of attempts to generate a valid answer.",
    )

    args = parser.parse_args()

    from drug_replace import (
        load_usmle_dataset,
        load_drug_map,
        format_and_flag_questions,
        filter_questions_by_flag,
        replace_keywords,
    )

    # Load the model
    model, tokenizer = load_model()

    # Load and process dataset
    dataset = load_usmle_dataset(args.dataset_name, args.split)
    keyword_map = load_drug_map(args.csv_path)
    formatted_questions = format_and_flag_questions(dataset, keyword_map.keys())
    filtered_questions = filter_questions_by_flag(formatted_questions)
    replaced_questions = replace_keywords(filtered_questions, keyword_map)

    # Evaluate both filtered and replaced datasets
    filtered_eval_results = evaluate_multiple_choice_questions(
        model,
        tokenizer,
        filtered_questions,
        os.path.join(args.output_dir, "filtered_results.json"),
        args.ntries,
    )
    replaced_eval_results = evaluate_multiple_choice_questions(
        model,
        tokenizer,
        replaced_questions,
        os.path.join(args.output_dir, "replaced_results.json"),
        args.ntries,
    )

    # Output evaluation results
    print("Filtered Questions Evaluation Results:")
    print(filtered_eval_results)
    print("Replaced Questions Evaluation Results:")
    print(replaced_eval_results)
