import pandas as pd
import os


def calculate_accuracy(correct_answer, inferred_answer, options):
    if (
        pd.isna(inferred_answer) or inferred_answer == ""
    ):  # Check if 'inferred_answer' is empty or NaN
        return "N/A"  # Return "N/A" for not available data
    else:
        try:
            inferred_answer = int(inferred_answer)
            answer_position = options.index(correct_answer) + 1  # 1-indexed
            return "Correct" if inferred_answer == answer_position else "Incorrect"
        except ValueError:
            return "N/A"


def process_general_knowledge(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    temperatures = df["temperature"].unique()
    temperature_accuracy_summaries = {}  # Store accuracy summaries for each temperature

    for temp in temperatures:
        temp_df = df[df["temperature"] == temp]

        # Calculate accuracy
        temp_df["brand_accuracy"] = temp_df.apply(
            lambda row: calculate_accuracy(
                row["correct_answer_brand"],
                row["inferred_brand_answer"],
                row["options_brand"],
            ),
            axis=1,
        )

        temp_df["preferred_accuracy"] = temp_df.apply(
            lambda row: calculate_accuracy(
                row["correct_answer_preferred"],
                row["inferred_preferred_answer"],
                row["options_preferred"],
            ),
            axis=1,
        )

        # Creating a summary DataFrame for brand and preferred accuracies for the current temperature
        accuracy_summary = {
            "Temperature": temp,
            "Type": ["Brand", "Preferred"],
            "Correct": [
                (temp_df["brand_accuracy"] == "Correct").sum(),
                (temp_df["preferred_accuracy"] == "Correct").sum(),
            ],
            "Incorrect": [
                (temp_df["brand_accuracy"] == "Incorrect").sum(),
                (temp_df["preferred_accuracy"] == "Incorrect").sum(),
            ],
            "Not Available": [
                (temp_df["brand_accuracy"] == "N/A").sum(),
                (temp_df["preferred_accuracy"] == "N/A").sum(),
            ],
        }

        accuracy_summary_df = pd.DataFrame(accuracy_summary)
        # Store the summary DataFrame in the dictionary with a key representing the temperature
        temperature_accuracy_summaries[temp] = accuracy_summary_df

        # Display the accuracy summary DataFrame for current temperature
        print(f"\nAccuracy Summary for Temperature {temp}:")
        print(accuracy_summary_df)

    # Concatenate all the individual accuracy summary DataFrames into one
    final_results_summary = pd.concat(temperature_accuracy_summaries.values())

    # Reset the index
    final_results_summary = final_results_summary.reset_index(drop=True)

    # Calculate accuracy as a percentage without considering 'Not Available' (N/A) answers
    final_results_summary["Accuracy w/o N/A (%)"] = (
        final_results_summary["Correct"]
        / (final_results_summary["Correct"] + final_results_summary["Incorrect"])
        * 100
    )

    # Calculate accuracy as a percentage with considering 'Not Available' (N/A) answers
    final_results_summary["Accuracy w/ N/A (%)"] = (
        final_results_summary["Correct"]
        / (
            final_results_summary["Correct"]
            + final_results_summary["Incorrect"]
            + final_results_summary["Not Available"]
        )
        * 100
    )

    # Display the final results summary
    print("Final Results Summary:")
    print(final_results_summary)

    # Save the final results summary to a CSV file
    final_results_summary.to_csv(
        os.path.join(output_dir, "general_knowledge_accuracy_summary.csv"), index=False
    )

    return final_results_summary
