import pandas as pd
import json
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


def load_parquet_files():
    generic_df = pd.read_parquet(
        "data/questions/drugs_contraindications_generic.parquet"
    )
    brand_df = pd.read_parquet("data/questions/drugs_contraindications_brand.parquet")
    return generic_df, brand_df


def load_api_responses(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def find_correct_index(options, correct_answer):
    return options.index(correct_answer) + 1  # 1-indexed


def is_correct(response, correct_index):
    try:
        response = int(response)
    except ValueError:
        return False
    return response == correct_index


def process_responses(
    df: pd.DataFrame, responses: List[Dict], drug_type: str, model_name: str
):
    temperatures = ["0.0", "0.7", "1.0"]
    results = []

    print(df.columns)

    # Adding the correct index to the DataFrame
    df["correct_index"] = df.apply(
        lambda row: find_correct_index(
            [row["ending0"], row["ending1"], row["ending2"], row["ending3"]],
            row["ending" + str(row["label"])],
        ),
        axis=1,
    )

    for temp in temperatures:
        correct_col = f"correct_{temp}_{drug_type}_{model_name}"
        df[correct_col] = False

        for response in responses:
            custom_id_parts = response["custom_id"].split("_")
            if custom_id_parts[4] == temp:
                index = int(custom_id_parts[0])  # Extract index from custom_id
                if 0 <= index < len(df):
                    api_response = response["response"]["body"]["choices"][0][
                        "message"
                    ]["content"]
                    df.at[index, correct_col] = is_correct(
                        api_response[0], df.at[index, "correct_index"]
                    )

        # Calculate counts and accuracy
        correct_count = df[correct_col].sum()
        total_count = df[correct_col].count()
        accuracy = correct_count / total_count * 100 if total_count > 0 else 0

        results.append(
            {
                "Model": model_name,
                "Temperature": temp,
                "Type": drug_type,
                "Correct": correct_count,
                "Total": total_count,
                "Accuracy (%)": accuracy,
            }
        )

    return pd.DataFrame(results)


def plot_results(results_df: pd.DataFrame, output_dir: str):
    models = results_df["Model"].unique()
    temperatures = sorted(results_df["Temperature"].unique())
    drug_types = results_df["Type"].unique()

    colors = {"generic": "#4a7ba7", "brand": "#a77b4a"}  # Muted blue and orange

    for model in models:
        model_results = results_df[results_df["Model"] == model]

        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        x = np.arange(len(temperatures))
        width = 0.35

        for i, drug_type in enumerate(drug_types):
            subset = model_results[model_results["Type"] == drug_type]
            bars = ax.bar(
                x + i * width,
                subset["Accuracy (%)"],
                width,
                label=drug_type,
                color=colors[drug_type],
                edgecolor="black",
                linewidth=0.5,
            )

            # Add value labels on the bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_xlabel("Temperature", fontweight="bold", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=12)
        ax.set_title(
            f"Contraindication Accuracy by Temperature for {model}",
            fontweight="bold",
            fontsize=14,
        )
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(temperatures, fontweight="bold")
        ax.legend(title="Drug Type", title_fontsize=12, fontsize=10, frameon=False)
        ax.set_ylim(0, 100)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{model}_accuracy_all_temps.png")
        print(f"Saving plot to {plot_path}")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


def plot_models_comparison(results_df: pd.DataFrame, output_dir: str):
    temp_0_results = results_df[results_df["Temperature"] == "0.0"]
    models = temp_0_results["Model"].unique()
    drug_types = temp_0_results["Type"].unique()

    colors = {"generic": "#4a7ba7", "brand": "#a77b4a"}  # Muted blue and orange

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    x = np.arange(len(models))
    width = 0.35

    for i, drug_type in enumerate(drug_types):
        subset = temp_0_results[temp_0_results["Type"] == drug_type]
        bars = ax.bar(
            x + i * width,
            subset["Accuracy (%)"],
            width,
            label=drug_type,
            color=colors[drug_type],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Model", fontweight="bold", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=12)
    ax.set_title(
        "Contraindication Accuracy Comparison (Temperature 0.0)",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, fontweight="bold", rotation=45, ha="right")
    ax.legend(title="Drug Type", title_fontsize=12, fontsize=10, frameon=False)
    ax.set_ylim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "model_comparison_temp_0.png")
    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def main():
    output_dir = "results/cx_mcq"
    os.makedirs(output_dir, exist_ok=True)

    generic_df, brand_df = load_parquet_files()

    models = ["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o"]
    all_results = []

    for model in models:
        generic_responses = load_api_responses(
            f"data/api_responses/{model}/cx_generic_responses.jsonl"
        )
        brand_responses = load_api_responses(
            f"data/api_responses/{model}/cx_brand_responses.jsonl"
        )

        generic_results = process_responses(
            generic_df, generic_responses, "generic", model
        )
        brand_results = process_responses(brand_df, brand_responses, "brand", model)

        all_results.extend([generic_results, brand_results])

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # Display the final results summary
    print("Final Results Summary:")
    print(results_df)

    # Save the final results summary to a CSV file
    results_df.to_csv(
        os.path.join(output_dir, "all_models_accuracy_summary.csv"), index=False
    )

    # Create a bar plot for each model
    plot_results(results_df, output_dir)

    # Create a bar plot comparing all models at temperature 0
    plot_models_comparison(results_df, output_dir)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
