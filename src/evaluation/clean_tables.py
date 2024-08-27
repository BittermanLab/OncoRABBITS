import csv
import os
import glob
from collections import defaultdict
import pandas as pd


def read_csv(file_path):
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def process_accuracy_data(data):
    processed = {}
    for row in data:
        temp = row["Temperature"]
        type_ = row["Type"]
        accuracy = float(row["Accuracy (%)"])
        key = f"{temp}_{type_}"
        processed[key] = accuracy
    return processed


def create_accuracy_summary_table(models):
    headers = ["Model", "Temperature", "Type", "Accuracy (%)"]
    rows = []

    for model, data in models.items():
        for temp in ["0.0", "0.7", "1.0"]:
            for type_ in ["brand name", "preferred name"]:
                key = f"{temp}_{type_}"
                if key in data:
                    rows.append([model, temp, type_, f"{data[key]:.2f}"])

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
            "brand_effective": int(row1["brand_effective"])
            + int(row2["brand_effective"]),
            "preferred_effective": int(row1["preferred_effective"])
            + int(row2["preferred_effective"]),
            "brand_ineffective": int(row1["brand_ineffective"])
            + int(row2["brand_ineffective"]),
            "preferred_ineffective": int(row1["preferred_ineffective"])
            + int(row2["preferred_ineffective"]),
            "brand_safe": int(row1["brand_safe"]) + int(row2["brand_safe"]),
            "preferred_safe": int(row1["preferred_safe"]) + int(row2["preferred_safe"]),
            "brand_unsafe": int(row1["brand_unsafe"]) + int(row2["brand_unsafe"]),
            "preferred_unsafe": int(row1["preferred_unsafe"])
            + int(row2["preferred_unsafe"]),
            "brand_has side effects": int(row1["brand_has side effects"])
            + int(row2["brand_has side effects"]),
            "preferred_has side effects": int(row1["preferred_has side effects"])
            + int(row2["preferred_has side effects"]),
            "brand_side effect free": int(row1["brand_side effect free"])
            + int(row2["brand_side effect free"]),
            "preferred_side effect free": int(row1["preferred_side effect free"])
            + int(row2["preferred_side effect free"]),
            "same_medication_count": int(row1["same_medication_count"])
            + int(row2["same_medication_count"]),
        }
        summed_data.append(summed_row)
    return summed_data


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


def process_sentiment_data(model_name):
    file_paths = [
        f"results/{model_name}/sentiment/summary_sentiment_question_about_{model_name}.csv",
        f"results/{model_name}/sentiment/summary_sentiment_question_patient_{model_name}.csv",
        f"results/{model_name}/sentiment/summary_sentiment_question_physician_{model_name}.csv",
    ]

    combined_data = defaultdict(dict)

    for file_path in file_paths:
        data = read_csv(file_path)
        question_type = file_path.split("_")[
            -2
        ]  # Extract 'about', 'patient', or 'physician'

        for row in data:
            string_type = row["string_type"]
            for temp in ["0.0", "0.7", "1.0"]:
                key = f"{question_type}_{temp}"
                # Round the sentiment value to 2 decimal places
                combined_data[string_type][
                    key
                ] = f"{float(row[f'sentiment_response_{temp}']):.2f}"

    return combined_data


def create_combined_sentiment_table(models):
    headers = [
        "Model",
        "String Type",
        "Question Type",
        "Temp 0.0",
        "Temp 0.7",
        "Temp 1.0",
    ]
    rows = []

    for model in models:
        data = process_sentiment_data(model)
        for string_type, values in data.items():
            for question_type in ["about", "patient", "physician"]:
                row = [
                    model,
                    string_type,
                    question_type,
                    values[f"{question_type}_0.0"],
                    values[f"{question_type}_0.7"],
                    values[f"{question_type}_1.0"],
                ]
                rows.append(row)

    return headers, rows


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

    list_preference_markdown = generate_markdown_table(
        list_preference_headers, list_preference_rows
    )
    print("\nList Preference Summary Table (Markdown):")
    print(list_preference_markdown)

    # Process sentiment data
    sentiment_models = ["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o"]
    sentiment_headers, sentiment_rows = create_combined_sentiment_table(
        sentiment_models
    )
    save_to_csv("sentiment_summary_table.csv", sentiment_headers, sentiment_rows)

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
