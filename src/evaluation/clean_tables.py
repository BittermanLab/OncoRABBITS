import csv
import os
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats

from cx_utils import cx_evaluation_main


def read_csv(file_path):
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def process_accuracy_data(data):
    processed = defaultdict(dict)

    for row in data:
        if row["Temperature"] != "Overall":
            temp = row["Temperature"]
            type_ = row["Type"]
            key = f"{temp}_{type_}"
            processed[key] = {
                "Accuracy (%)": float(row["Accuracy (%)"]),
                "Standard Error": row["Standard Error"],
                "T-Statistic": row["T-Statistic"],
                "P-Value": row["P-Value"],
            }
            # make sure temp, acc, se, t-stat, p-value are floats
            processed[key]["Accuracy (%)"] = float(processed[key]["Accuracy (%)"])
            processed[key]["Standard Error"] = float(processed[key]["Standard Error"])
            processed[key]["T-Statistic"] = float(processed[key]["T-Statistic"])
            processed[key]["P-Value"] = float(processed[key]["P-Value"])

    return processed


def create_accuracy_summary_table(models):
    headers = [
        "Model",
        "Temperature",
        "Brand Accuracy (%) [SE]",
        "Generic Accuracy (%) [SE]",
        "T-statistic",
        "P-value",
    ]
    rows = []

    for model, data in models.items():
        for temp in ["0.0", "0.7", "1.0"]:
            brand_key = f"{temp}_brand name"
            preferred_key = f"{temp}_preferred name"

            if brand_key in data and preferred_key in data:
                brand_data = data[brand_key]
                preferred_data = data[preferred_key]

                row = [
                    model,
                    temp,
                    f"{brand_data['Accuracy (%)']:.2f} [{brand_data['Standard Error']:.2f}]",
                    f"{preferred_data['Accuracy (%)']:.2f} [{preferred_data['Standard Error']:.2f}]",
                    f"{brand_data['T-Statistic']:.4f}",
                    f"{brand_data['P-Value']:.4f}",
                ]
                rows.append(row)

    # Sort rows by model and temperature
    rows.sort(key=lambda x: (x[0], x[1]))

    footnotes = [
        "SE: Standard Error",
        "T-statistic: Measure of the difference between brand and generic accuracy",
        "P-value: Statistical significance of the difference (p < 0.05 is considered significant)",
    ]

    return headers, rows, footnotes


def join_list_preference_tables(prompt1_data, prompt2_data):
    joined_data = []
    for row1, row2 in zip(prompt1_data, prompt2_data):
        joined_row = {
            "temperature": row1["temperature"],
            "engine": row1["engine"],
            "brand_effective": row1["brand_effective"],
            "preferred_effective": row1["preferred_effective"],
            "brand_ineffective": row1["brand_ineffective"],
            "preferred_ineffective": row1["preferred_ineffective"],
            "brand_safe": row1["brand_safe"],
            "preferred_safe": row1["preferred_safe"],
            "brand_unsafe": row1["brand_unsafe"],
            "preferred_unsafe": row1["preferred_unsafe"],
            "brand_has side effects": row2["brand_has side effects"],
            "preferred_has side effects": row2["preferred_has side effects"],
            "brand_side effect free": row2["brand_side effect free"],
            "preferred_side effect free": row2["preferred_side effect free"],
            "same_medication_count": row1["same_medication_count"],
        }
        joined_data.append(joined_row)
    return joined_data


def sum_list_preference_tables(prompt1_data, prompt2_data):
    summed_data = []
    for row1, row2 in zip(prompt1_data, prompt2_data):
        summed_row = {
            "temperature": row1["temperature"],
            "engine": row1["engine"],
            "effective": (
                int(row1["brand_effective"]) + int(row2["brand_effective"]),
                int(row1["preferred_effective"]) + int(row2["preferred_effective"]),
                float(row1["effective_odds_ratio"]),
                float(row1["effective_p_value"]),
            ),
            "ineffective": (
                int(row1["brand_ineffective"]) + int(row2["brand_ineffective"]),
                int(row1["preferred_ineffective"]) + int(row2["preferred_ineffective"]),
                float(row1["ineffective_odds_ratio"]),
                float(row1["ineffective_p_value"]),
            ),
            "safe": (
                int(row1["brand_safe"]) + int(row2["brand_safe"]),
                int(row1["preferred_safe"]) + int(row2["preferred_safe"]),
                float(row1["safe_odds_ratio"]),
                float(row1["safe_p_value"]),
            ),
            "unsafe": (
                int(row1["brand_unsafe"]) + int(row2["brand_unsafe"]),
                int(row1["preferred_unsafe"]) + int(row2["preferred_unsafe"]),
                float(row1["unsafe_odds_ratio"]),
                float(row1["unsafe_p_value"]),
            ),
            "has side effects": (
                int(row1["brand_has side effects"])
                + int(row2["brand_has side effects"]),
                int(row1["preferred_has side effects"])
                + int(row2["preferred_has side effects"]),
                float(row1["has side effects_odds_ratio"]),
                float(row1["has side effects_p_value"]),
            ),
            "side effect free": (
                int(row1["brand_side effect free"])
                + int(row2["brand_side effect free"]),
                int(row1["preferred_side effect free"])
                + int(row2["preferred_side effect free"]),
                float(row1["side effect free_odds_ratio"]),
                float(row1["side effect free_p_value"]),
            ),
            "same_medication_count": int(row1["same_medication_count"])
            + int(row2["same_medication_count"]),
        }
        summed_data.append(summed_row)
    return summed_data


def format_cell(data):
    brand, preferred, odds_ratio, p_value = data
    significance = "*" if p_value < 0.05 else ""
    return f"{brand}/{preferred} ({odds_ratio:.2f}{significance})"


def process_list_preference_model(model_name):
    base_path = f"results/{model_name}/list_preference/"
    prompt1_file = (
        f"{base_path}list_preference_prompt1_{model_name}_aggregated_counts_list.csv"
    )
    prompt2_file = (
        f"{base_path}list_preference_prompt2_{model_name}_aggregated_counts_list.csv"
    )

    prompt1_data = read_csv(prompt1_file)
    prompt2_data = read_csv(prompt2_file)

    return sum_list_preference_tables(prompt1_data, prompt2_data)


def save_to_csv(filename, headers, rows):
    if not os.path.exists("results/tables"):
        os.makedirs("results/tables")
    with open(f"results/tables/{filename}", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def generate_markdown_table(headers, rows, footnotes=None):
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "|" + "|".join(["---" for _ in headers]) + "|\n"

    for row in rows:
        markdown_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    if footnotes:
        markdown_table += "\n"
        for note in footnotes:
            markdown_table += f"- {note}\n"

    return markdown_table


def generate_list_preference_markdown_table(data):
    headers = [
        "Model",
        "Temp",
        "Effective\nBrand/Generic (OR)",
        "Ineffective\nBrand/Generic (OR)",
        "Safe\nBrand/Generic (OR)",
        "Unsafe\nBrand/Generic (OR)",
        "Has Side Effects\nBrand/Generic (OR)",
        "Side Effect Free\nBrand/Generic (OR)",
        "Same Med",
    ]
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in data:
        formatted_row = [
            row["engine"],
            row["temperature"],
            format_cell(row["effective"]),
            format_cell(row["ineffective"]),
            format_cell(row["safe"]),
            format_cell(row["unsafe"]),
            format_cell(row["has side effects"]),
            format_cell(row["side effect free"]),
            str(row["same_medication_count"]),
        ]
        markdown += "| " + " | ".join(formatted_row) + " |\n"

    markdown += "\n* Asterisk indicates statistical significance (p < 0.05)\n"
    markdown += "OR: Odds Ratio. Values > 1 indicate higher odds for brand names, < 1 for generic names.\n"
    markdown += "Each cell format: Brand count / Generic count (Odds Ratio)"
    return markdown


def process_sentiment_data(model_name):
    file_path = f"results/{model_name}/sentiment/sentiment_question_about_{model_name}_sentiment_summary.csv"
    data = pd.read_csv(file_path)

    combined_data = defaultdict(lambda: defaultdict(dict))

    for _, row in data.iterrows():
        temp = row["Temperature"]
        combined_data[temp] = {
            "Brand Mean": row["Brand Mean Sentiment"],
            "Generic Mean": row["Preferred Mean Sentiment"],
            "Chi-square": row["Chi-square Statistic"],
            "p-value": row["Chi-square p-value"],
        }

    return combined_data


def create_combined_sentiment_table(models):
    headers = [
        "Model",
        "Temperature",
        "Brand Mean",
        "Generic Mean",
        "Chi-square",
        "p-value",
    ]
    rows = []

    for model in models:
        data = process_sentiment_data(model)
        for temp, metrics in data.items():
            row = [
                model,
                temp,
                f"{metrics['Brand Mean']:.4f}",
                f"{metrics['Generic Mean']:.4f}",
                f"{metrics['Chi-square']:.4f}",
                f"{metrics['p-value']:.4f}",
            ]
            rows.append(row)

    footnotes = [
        "Brand Mean and Generic Mean: Average sentiment scores (higher values indicate more positive sentiment)",
        "Chi-square: Measure of the difference between brand and generic sentiment distributions",
        "p-value: Statistical significance of the difference (p < 0.05 is considered significant)",
    ]

    return headers, rows, footnotes


def print_table(headers, rows, footnotes=None):
    # Calculate column widths
    col_widths = [
        max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))
    ]

    # Print headers
    header_row = " | ".join(
        f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))
    )
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for row in rows:
        print(" | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))))

    # Print footnotes
    if footnotes:
        print("\nFootnotes:")
        for note in footnotes:
            print(f"- {note}")


def load_and_process_csv(file_path, task, model):
    df = pd.read_csv(file_path)
    df["task"] = task
    df["model"] = model
    return df


def process_detection_data(df, stats_df):
    df = df.rename(columns={"average_score": "mean_score", "std_dev": "std_score"})
    df = df[
        [
            "task",
            "model",
            "type",
            "temperature",
            "mean_score",
            "median_score",
            "std_score",
        ]
    ]

    # Merge with stats data
    df = pd.merge(df, stats_df, on="temperature", how="left")

    return df


def process_differential_data(df, stats_df):

    df_melted = pd.melt(
        df,
        id_vars=["type", "task", "model"],
        var_name="metric",
        value_name="percentage",
    )

    df_melted[["event_type", "temp", "stat"]] = df_melted["metric"].str.extract(
        r"(\w+)_percentage_(\d+\.\d+)_(\w+)"
    )

    df_pivot = df_melted.pivot_table(
        values="percentage",
        index=["task", "model", "type", "temp", "event_type", "stat"],
        columns="event_type",
        aggfunc="first",
    ).reset_index()

    df_pivot.columns.name = None
    df_pivot = df_pivot.rename(columns={"temp": "temperature"})

    # Rename columns to ensure uniqueness
    df_pivot = df_pivot.rename(
        columns={
            "drug": "drug_percentage",
            "general": "general_percentage",
            "irae": "irae_percentage",
        }
    )

    # Convert 'temperature' to float in both dataframes
    df_pivot["temperature"] = df_pivot["temperature"].astype(float)
    stats_df["temperature"] = stats_df["temperature"].astype(float)

    # Ensure 'event_type' values match between dataframes
    df_pivot["event_type"] = df_pivot["event_type"].str.lower()
    stats_df["event_type"] = stats_df["event_type"].str.lower()

    # Merge with stats data
    df_pivot = pd.merge(
        df_pivot,
        stats_df,
        on=["temperature", "event_type"],
        how="left",
        suffixes=("", "_stats"),
    )
    duplicate_columns = df_pivot.columns[df_pivot.columns.duplicated(keep="first")]
    # Drop duplicate columns
    df_pivot = df_pivot.drop(columns=duplicate_columns)

    return df_pivot


def create_irae_summary_tables(input_dir):
    models = ["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o"]
    tasks = ["detection", "differential"]

    detection_data = []
    differential_data = []

    for model in models:
        # Process detection data
        detection_file = os.path.join(
            input_dir,
            "irae",
            model,
            "irae_detection",
            f"irae_detection_{model}_results.csv",
        )
        detection_stats_file = os.path.join(
            input_dir,
            "irae",
            model,
            "irae_detection",
            f"irae_detection_{model}_stats.csv",
        )
        if os.path.exists(detection_file) and os.path.exists(detection_stats_file):
            df = load_and_process_csv(detection_file, "detection", model)
            stats_df = pd.read_csv(detection_stats_file)
            processed_df = process_detection_data(df, stats_df)
            detection_data.append(processed_df)
        else:
            print(f"Warning: No detection file or stats file found for {model}")

        # Process differential data
        differential_pattern = os.path.join(
            input_dir,
            "irae",
            model,
            "differential",
            f"differential_{model}_summary.csv",
        )
        differential_stats_pattern = os.path.join(
            input_dir,
            "irae",
            model,
            "differential",
            f"differential_{model}_stats.csv",
        )
        differential_files = glob.glob(differential_pattern)
        differential_stats_files = glob.glob(differential_stats_pattern)
        if differential_files and differential_stats_files:
            for file, stats_file in zip(differential_files, differential_stats_files):
                df = load_and_process_csv(file, "differential", model)
                stats_df = pd.read_csv(stats_file)
                processed_df = process_differential_data(df, stats_df)
                differential_data.append(processed_df)
        else:
            print(f"Warning: No differential file or stats file found for {model}")

    if not detection_data and not differential_data:
        raise ValueError(
            "No iRAE data found. Check the input directory and file patterns."
        )

    detection_summary = pd.DataFrame()
    differential_summary = pd.DataFrame()

    if detection_data:
        detection_summary = pd.concat(detection_data, ignore_index=True)
        detection_summary = detection_summary.sort_values(["model", "temperature"])

        # Pivot the detection summary to have brand and generic side by side
        detection_summary = detection_summary.pivot(
            index=["model", "temperature"],
            columns="type",
            values=[
                "mean_score",
                "std_score",
                "t_statistic",
                "p_value",
            ],
        )
        detection_summary.columns = [
            f"{col[1]}_{col[0]}" for col in detection_summary.columns
        ]
        detection_summary = detection_summary.reset_index()

        # Reorder columns
        column_order = [
            "model",
            "temperature",
            "brand_mean_score",
            "generic_mean_score",
            "brand_std_score",
            "generic_std_score",
            "brand_t_statistic",
            "generic_t_statistic",
            "brand_p_value",
            "generic_p_value",
        ]
        detection_summary = detection_summary[column_order]

        # Round numeric columns to 2 decimal places
        numeric_columns = detection_summary.columns.drop(["model", "temperature"])
        detection_summary[numeric_columns] = detection_summary[numeric_columns].round(2)

        # Format mean and stderr columns
        for prefix in ["brand", "generic"]:
            detection_summary[f"{prefix}_formatted"] = detection_summary.apply(
                lambda row: f"{row[f'{prefix}_mean_score']:.2f} ({row[f'{prefix}_std_score']:.2f})",
                axis=1,
            )

        # Create t-statistic and p-value column
        detection_summary["t_stat_p_value"] = detection_summary.apply(
            lambda row: f"{row['brand_t_statistic']:.2f} ({row['brand_p_value']:.3f})",
            axis=1,
        )

        # Final column order
        final_columns = [
            "model",
            "temperature",
            "brand_formatted",
            "generic_formatted",
            "t_stat_p_value",
        ]
        detection_summary = detection_summary[final_columns]

    if differential_data:
        differential_summary = pd.concat(differential_data, ignore_index=True)

        # Pivot the table to create the new format
        pivoted = differential_summary.pivot_table(
            index=["model", "temperature"],  # Include 'model' in the index
            columns=["type", "event_type", "stat"],
            values=["t_statistic", "p_value"],
            aggfunc="first",
        )

        # Flatten column names
        pivoted.columns = [
            f"{col[1]}_{col[2]}_{col[3]}_{col[0]}" for col in pivoted.columns
        ]

        # Reset index to make 'model' and 'temperature' regular columns
        pivoted = pivoted.reset_index()

        # Round numeric values to 2 decimal places
        numeric_columns = pivoted.columns.drop(["model", "temperature"])
        pivoted[numeric_columns] = pivoted[numeric_columns].round(2)

        # Combine t_statistic and p_value into a single column
        new_columns = ["model", "temperature"]
        for base_col in [
            "brand_drug",
            "brand_general",
            "brand_irae",
            "generic_drug",
            "generic_general",
            "generic_irae",
        ]:
            mean_col = f"{base_col}_mean_t_statistic"
            std_col = f"{base_col}_std_t_statistic"
            t_stat_col = f"{base_col}_mean_t_statistic"
            p_value_col = f"{base_col}_mean_p_value"

            if all(
                col in pivoted.columns
                for col in [mean_col, std_col, t_stat_col, p_value_col]
            ):
                pivoted[f"{base_col}_mean_std"] = pivoted.apply(
                    lambda row: f"{row[mean_col]:.2f} ({row[std_col]:.2f})",
                    axis=1,
                )
                pivoted[f"{base_col}_t_p"] = pivoted.apply(
                    lambda row: f"{row[t_stat_col]:.2f} ({row[p_value_col]:.3f})",
                    axis=1,
                )
                new_columns.extend([f"{base_col}_mean_std", f"{base_col}_t_p"])

        # Keep only the new combined columns
        pivoted = pivoted[new_columns]

        # Replace NaN with '-'
        pivoted = pivoted.fillna("-")

        # Sort by model and temperature
        pivoted = pivoted.sort_values(["model", "temperature"])

        differential_summary = pivoted

    return detection_summary, differential_summary


def main():
    # Process accuracy data
    model_files = {
        "gpt-4-turbo": "results/gpt-4-turbo/general_knowledge/gpt-4-turbo_accuracy_summary.csv",
        "gpt-3.5-turbo-0125": "results/gpt-3.5-turbo-0125/general_knowledge/gpt-3.5-turbo-0125_accuracy_summary.csv",
        "gpt-4o": "results/gpt-4o/general_knowledge/gpt-4o_accuracy_summary.csv",
    }

    accuracy_models_data = {}
    for model, file_path in model_files.items():
        csv_data = read_csv(file_path)
        accuracy_models_data[model] = process_accuracy_data(csv_data)

    accuracy_headers, accuracy_rows, accuracy_footnotes = create_accuracy_summary_table(
        accuracy_models_data
    )
    save_to_csv("accuracy_summary_table.csv", accuracy_headers, accuracy_rows)
    accuracy_markdown = generate_markdown_table(
        accuracy_headers, accuracy_rows, accuracy_footnotes
    )
    print("Accuracy Summary Table (Markdown):")
    print(accuracy_markdown)

    # Process list preference data
    list_preference_models = ["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o"]
    list_preference_data = []

    for model in list_preference_models:
        list_preference_data.extend(process_list_preference_model(model))

    list_preference_data.sort(key=lambda x: (x["engine"], float(x["temperature"])))

    list_preference_headers = list_preference_data[0].keys()
    list_preference_rows = [list(row.values()) for row in list_preference_data]
    save_to_csv(
        "list_preference_summary_table.csv",
        list_preference_headers,
        list_preference_rows,
    )

    list_preference_markdown = generate_list_preference_markdown_table(
        list_preference_data
    )
    print("\nList Preference Summary Table (Markdown):")
    print(list_preference_markdown)

    # Process sentiment data
    sentiment_models = ["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o"]
    sentiment_headers, sentiment_rows, sentiment_footnotes = (
        create_combined_sentiment_table(sentiment_models)
    )
    save_to_csv("sentiment_summary_table.csv", sentiment_headers, sentiment_rows)
    sentiment_markdown = generate_markdown_table(
        sentiment_headers, sentiment_rows, sentiment_footnotes
    )
    print("\nSentiment Summary Table (Markdown):")
    print(sentiment_markdown)

    # Process iRAE data
    irae_input_dir = "results"  # This should be the path to the directory containing the "irae" folder
    try:
        detection_summary, differential_summary = create_irae_summary_tables(
            irae_input_dir
        )

        if not detection_summary.empty:
            detection_summary.to_csv(
                "results/tables/irae_detection_summary_table.csv", index=False
            )
            detection_headers = [
                "Model",
                "Temperature",
                "Brand Mean (SE)",
                "Generic Mean (SE)",
                "T-statistic (p-value)",
            ]
            detection_rows = detection_summary.values.tolist()
            detection_footnotes = [
                "SE: Standard Error",
                "T-statistic: Measure of the difference between brand and generic scores",
                "p-value: Statistical significance of the difference (p < 0.05 is considered significant)",
            ]
            detection_markdown = generate_markdown_table(
                detection_headers, detection_rows, detection_footnotes
            )
            print("\niRAE Detection Summary Table (Markdown):")
            print(detection_markdown)
        else:
            print("No detection data available.")

        if not differential_summary.empty:
            differential_summary.to_csv(
                "results/tables/irae_differential_summary_table.csv", index=False
            )
            differential_headers = [
                "Model",
                "Temperature",
                "Brand Drug Mean (Std)",
                "Brand Drug T (p)",
                "Brand General Mean (Std)",
                "Brand General T (p)",
                "Brand iRAE Mean (Std)",
                "Brand iRAE T (p)",
                "Generic Drug Mean (Std)",
                "Generic Drug T (p)",
                "Generic General Mean (Std)",
                "Generic General T (p)",
                "Generic iRAE Mean (Std)",
                "Generic iRAE T (p)",
            ]
            differential_rows = differential_summary.values.tolist()
            differential_footnotes = [
                "Mean: Average t-statistic for the comparison",
                "Std: Standard deviation of the t-statistic",
                "p-value: Statistical significance of the difference (p < 0.05 is considered significant)",
                "iRAE: immune-related Adverse Event",
            ]
            differential_markdown = generate_markdown_table(
                differential_headers, differential_rows, differential_footnotes
            )
            print("\niRAE Differential Summary Table (Markdown):")
            print(differential_markdown)
        else:
            print("No differential data available.")

    except ValueError as e:
        print(f"Error processing iRAE data: {e}")

    # Process CX data
    summary_table = cx_evaluation_main()
    cx_footnotes = [
        "SE: Standard Error",
        "T-statistic: Measure of the difference between brand and generic scores",
        "p-value: Statistical significance of the difference (p < 0.05 is considered significant)",
    ]
    summary_markdown = generate_markdown_table(
        summary_table.columns, summary_table.values, cx_footnotes
    )
    print("\nCX Summary Table (Markdown):")
    print(summary_markdown)


if __name__ == "__main__":
    main()
