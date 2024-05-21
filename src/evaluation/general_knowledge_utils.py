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
        except (ValueError, IndexError):
            return "N/A"


def process_general_knowledge(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    temperatures = ["0.0", "0.7", "2.0"]
    temperature_accuracy_summaries = {}

    for temp in temperatures:
        temp_brand_col = f"response_{temp}"
        temp_preferred_col = f"response_{temp}"

        print(f"Example options and correct answer for temperature {temp}:")
        print(df[["options", "correct_answer", temp_brand_col]].head(1))

        # Calculate accuracy
        df[f"brand_accuracy_{temp}"] = df.apply(
            lambda row: (
                calculate_accuracy(
                    row["correct_answer"],
                    row[temp_brand_col],
                    eval(
                        row["options"]
                    ),  # Evaluate the string representation of the list
                )
                if row["string_type"] == "brand name"
                else "N/A"
            ),
            axis=1,
        )

        df[f"preferred_accuracy_{temp}"] = df.apply(
            lambda row: (
                calculate_accuracy(
                    row["correct_answer"],
                    row[temp_preferred_col],
                    eval(
                        row["options"]
                    ),  # Evaluate the string representation of the list
                )
                if row["string_type"] == "preferred name"
                else "N/A"
            ),
            axis=1,
        )

        # Creating a summary DataFrame for brand and preferred accuracies for the current temperature
        accuracy_summary = {
            "Temperature": temp,
            "Type": ["Brand", "Preferred"],
            "Correct": [
                (df[f"brand_accuracy_{temp}"] == "Correct").sum(),
                (df[f"preferred_accuracy_{temp}"] == "Correct").sum(),
            ],
            "Incorrect": [
                (df[f"brand_accuracy_{temp}"] == "Incorrect").sum(),
                (df[f"preferred_accuracy_{temp}"] == "Incorrect").sum(),
            ],
            "Not Available": [
                (df[f"brand_accuracy_{temp}"] == "N/A").sum(),
                (df[f"preferred_accuracy_{temp}"] == "N/A").sum(),
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
        os.path.join(output_dir, "general_knowledge/accuracy_summary.csv"), index=False
    )

    return final_results_summary
