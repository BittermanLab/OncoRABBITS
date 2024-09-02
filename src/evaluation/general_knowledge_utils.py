import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def find_correct_index(options, correct_answer):
    options = ast.literal_eval(options)
    return options.index(correct_answer) + 1  # 1-indexed


def is_correct(response, correct_index):
    try:
        response = int(float(response))  # Convert to float first, then to int
    except ValueError:
        print(response)
        print(correct_index)
        return False
    return response == correct_index


def process_general_knowledge(
    df: pd.DataFrame, output_dir: str, model_name: str
) -> pd.DataFrame:
    temperatures = ["0.0", "0.7", "1.0"]
    results = []
    overall_correct = 0
    overall_total = 0

    df["correct_index"] = df.apply(
        lambda row: find_correct_index(row["options"], row["correct_answer"]), axis=1
    )

    for temp in temperatures:
        for string_type in ["brand name", "preferred name"]:
            correct_col = f"correct_{temp}_{string_type.replace(' ', '_')}"
            response_col = f"response_{temp}"

            df[correct_col] = df.apply(
                lambda row: (
                    is_correct(row[response_col], row["correct_index"])
                    if row["string_type"] == string_type
                    else False
                ),
                axis=1,
            )

            filtered_df = df[df["string_type"] == string_type]
            correct_count = filtered_df[correct_col].sum()
            total_count = filtered_df[
                correct_col
            ].count()  # Reverted to original counting method

            overall_correct += correct_count
            overall_total += total_count

            accuracy = correct_count / total_count * 100 if total_count > 0 else 0
            p = accuracy / 100
            standard_error = (
                np.sqrt((p * (1 - p)) / total_count) * 100 if total_count > 0 else 0
            )

            results.append(
                {
                    "Temperature": temp,
                    "Type": string_type,
                    "Correct": correct_count,
                    "Total": total_count,
                    "Accuracy (%)": accuracy,
                    "Standard Error": standard_error,
                }
            )

    # Calculate overall metrics
    overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0
    overall_p = overall_accuracy / 100
    overall_se = (
        np.sqrt((overall_p * (1 - overall_p)) / overall_total) * 100
        if overall_total > 0
        else 0
    )

    results.append(
        {
            "Temperature": "Overall",
            "Type": "All",
            "Correct": overall_correct,
            "Total": overall_total,
            "Accuracy (%)": overall_accuracy,
            "Standard Error": overall_se,
        }
    )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Perform t-tests comparing brand name and preferred name for each temperature
    for temp in temperatures:
        brand_data = results_df[
            (results_df["Temperature"] == temp) & (results_df["Type"] == "brand name")
        ]
        preferred_data = results_df[
            (results_df["Temperature"] == temp)
            & (results_df["Type"] == "preferred name")
        ]

        brand_accuracy = brand_data["Accuracy (%)"].values[0]
        preferred_accuracy = preferred_data["Accuracy (%)"].values[0]
        brand_total = brand_data["Total"].values[0]
        preferred_total = preferred_data["Total"].values[0]

        # Calculate standard errors
        brand_se = (
            np.sqrt((brand_accuracy / 100 * (1 - brand_accuracy / 100)) / brand_total)
            * 100
        )
        preferred_se = (
            np.sqrt(
                (preferred_accuracy / 100 * (1 - preferred_accuracy / 100))
                / preferred_total
            )
            * 100
        )

        # Calculate t-statistic and p-value
        t_stat = (brand_accuracy - preferred_accuracy) / np.sqrt(
            brand_se**2 + preferred_se**2
        )
        df = brand_total + preferred_total - 2  # Degrees of freedom
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        # Add t-test results to results_df
        results_df.loc[results_df["Temperature"] == temp, "T-Statistic"] = t_stat
        results_df.loc[results_df["Temperature"] == temp, "P-Value"] = p_value

    # Display the final results summary
    print("Final Results Summary:")
    print(results_df)

    # Save the final results summary to a CSV file
    results_df.to_csv(
        os.path.join(output_dir, f"{model_name}_accuracy_summary.csv"),
        index=False,
    )

    return results_df


def plot_general_knowledge(output_dir: str, model_name: str):
    # Load the data
    final_results_summary = pd.read_csv(
        os.path.join(output_dir, f"general_knowledge/{model_name}_accuracy_summary.csv")
    )
    print(final_results_summary)

    # Plot
    temperatures = final_results_summary["Temperature"].unique()
    types = final_results_summary["Type"].unique()
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    index = np.arange(len(temperatures))

    brand_data = final_results_summary[final_results_summary["Type"] == "brand name"]
    preferred_data = final_results_summary[
        final_results_summary["Type"] == "preferred name"
    ]

    ax.bar(
        index, brand_data["Accuracy (%)"], bar_width, label="Brand Name", color="blue"
    )
    ax.bar(
        index + bar_width,
        preferred_data["Accuracy (%)"],
        bar_width,
        label="Preferred Name",
        color="orange",
    )

    ax.set_title("Accuracy by Temperature and Type")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(temperatures)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    plot_dir = os.path.join(output_dir, f"general_knowledge/plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = f"{plot_dir}/accuracy_all_temps.png"
    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path)
    plt.close()

    return


if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        "concept_code": [4] * 10,
        "string_type": ["brand name"] * 5 + ["preferred name"] * 5,
        "string": ["Verzenio"] * 5 + ["Abemaciclib"] * 5,
        "question": [
            "Which of the following is the brand name for the drug Abemaciclib?"
        ]
        * 5
        + ["Which of the following is the generic name for the drug Verzenio?"] * 5,
        "options": ["['Abemaciclib', 'Amoxi', 'Kitent', 'Verzenio']"] * 10,
        "correct_answer": ["Verzenio"] * 5 + ["Abemaciclib"] * 5,
        "response_0.0": ["4", "3", "4", "4", "1"] * 2,
        "response_0.7": ["4", "3", "4", "4", "2"] * 2,
        "response_1.0": ["4", "4", "4", "4", "3"] * 2,
    }

    df = pd.DataFrame(data)

    # Test the function
    output_dir = "test_output"
    model_name = "test_model"
    result = process_general_knowledge(df, output_dir, model_name)
    print("\nReturned DataFrame:")
    print(result)
