import csv
import os
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats


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
        "Brand Accuracy (%)",
        "Generic Accuracy (%)",
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
                    f"{brand_data['Accuracy (%)']:.2f}",
                    f"{preferred_data['Accuracy (%)']:.2f}",
                    f"{brand_data['T-Statistic']:.4f}",
                    f"{brand_data['P-Value']:.4f}",
                ]
                rows.append(row)

    # Sort rows by model and temperature
    rows.sort(key=lambda x: (x[0], x[1]))

    return headers, rows


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


def generate_markdown_table(headers, rows):
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "|" + "|".join(["---" for _ in headers]) + "|\n"

    for row in rows:
        markdown_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    return markdown_table


def generate_list_preference_markdown_table(data):
    headers = [
        "Model",
        "Temp",
        "Effective",
        "Ineffective",
        "Safe",
        "Unsafe",
        "Has Side Effects",
        "Side Effect Free",
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

    markdown += "\n* Asterisk indicates statistical significance (p < 0.05)"
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

    return headers, rows


def print_table(headers, rows):
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


def load_and_process_csv(file_path, task, model):
    df = pd.read_csv(file_path)
    df["task"] = task
    df["model"] = model
    return df


def process_detection_data(df):
    df = df.rename(columns={"average_score": "mean_score", "std_dev": "std_score"})
    return df[
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


def process_differential_data(df):
    df_melted = pd.melt(
        df,
        id_vars=["type", "task", "model"],
        var_name="metric",
        value_name="percentage",
    )

    df_melted[["event_type", "temp"]] = df_melted["metric"].str.extract(
        r"(\w+)_percentage_(\d+\.\d+)"
    )

    df_pivot = df_melted.pivot_table(
        values="percentage",
        index=["task", "model", "type", "temp"],
        columns="event_type",
        aggfunc="first",
    ).reset_index()

    df_pivot = df_pivot.rename(columns={"temp": "temperature"})

    return df_pivot


def create_irae_summary_tables(input_dir):
    models = ["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o"]
    tasks = ["detection", "differential"]

    detection_data = []
    differential_data = []

    for model in models:
        # Process detection data
        detection_file = os.path.join(
            input_dir, "irae", model, "irae_detection", "irae_detection_results.csv"
        )
        if os.path.exists(detection_file):
            df = load_and_process_csv(detection_file, "detection", model)
            processed_df = process_detection_data(df)
            detection_data.append(processed_df)
        else:
            print(f"Warning: No detection file found for {model}")

        # Process differential data
        differential_pattern = os.path.join(
            input_dir,
            "irae",
            model,
            "differential",
            f"differential_{model}_summary.csv",
        )
        differential_files = glob.glob(differential_pattern)
        if differential_files:
            for file in differential_files:
                df = load_and_process_csv(file, "differential", model)
                processed_df = process_differential_data(df)
                differential_data.append(processed_df)
        else:
            print(f"Warning: No differential file found for {model}")

    if not detection_data and not differential_data:
        raise ValueError(
            "No iRAE data found. Check the input directory and file patterns."
        )

    detection_summary = pd.DataFrame()
    differential_summary = pd.DataFrame()

    if detection_data:
        detection_summary = pd.concat(detection_data, ignore_index=True)
        detection_summary = detection_summary.sort_values(
            ["model", "type", "temperature"]
        )
        # Set numeric columns to 2 decimal places
        numeric_columns = ["mean_score", "median_score", "std_score"]
        detection_summary[numeric_columns] = detection_summary[numeric_columns].round(2)

    if differential_data:
        differential_summary = pd.concat(differential_data, ignore_index=True)
        differential_summary = differential_summary.sort_values(
            ["model", "type", "temperature"]
        )
        # Set numeric columns to 2 decimal places
        numeric_columns = ["drug", "general", "irae"]
        differential_summary[numeric_columns] = differential_summary[
            numeric_columns
        ].round(2)

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

    accuracy_headers, accuracy_rows = create_accuracy_summary_table(
        accuracy_models_data
    )
    save_to_csv("accuracy_summary_table.csv", accuracy_headers, accuracy_rows)
    accuracy_markdown = generate_markdown_table(accuracy_headers, accuracy_rows)
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
    sentiment_headers, sentiment_rows = create_combined_sentiment_table(
        sentiment_models
    )
    save_to_csv("sentiment_summary_table.csv", sentiment_headers, sentiment_rows)
    print("\nSentiment Summary Table:")
    print_table(sentiment_headers, sentiment_rows)

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
            detection_headers = detection_summary.columns.tolist()
            detection_rows = detection_summary.values.tolist()
            detection_markdown = generate_markdown_table(
                detection_headers, detection_rows
            )
            print("\niRAE Detection Summary Table (Markdown):")
            print(detection_markdown)
        else:
            print("No detection data available.")

        if not differential_summary.empty:
            differential_summary.to_csv(
                "results/tables/irae_differential_summary_table.csv", index=False
            )
            differential_headers = differential_summary.columns.tolist()
            differential_rows = differential_summary.values.tolist()
            differential_markdown = generate_markdown_table(
                differential_headers, differential_rows
            )
            print("\niRAE Differential Summary Table (Markdown):")
            print(differential_markdown)
        else:
            print("No differential data available.")

    except ValueError as e:
        print(f"Error processing iRAE data: {e}")


if __name__ == "__main__":
    main()
