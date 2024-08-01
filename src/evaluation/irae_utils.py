import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import ast


def process_irae_detection(df, output_dir, task_name, model):
    # Process irae detection data
    # Convert string responses to numeric values
    for temp in ["0.0", "0.7", "1.0"]:
        df[f"score_{temp}"] = pd.to_numeric(df[f"response_{temp}"], errors="coerce")

    # Calculate statistics
    results = []
    for temp in ["0.0", "0.7", "1.0"]:
        temp_stats = (
            df.groupby("type")[f"score_{temp}"]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        temp_stats["temperature"] = temp
        results.append(temp_stats)

    results_df = pd.concat(results, ignore_index=True)
    results_df = results_df.rename(
        columns={"mean": "average_score", "median": "median_score", "std": "std_dev"}
    )

    # Plot the results
    plot_mean_irae_detection(results_df, output_dir, task_name, model)

    # Plot the count distribution
    plot_irae_detection_counts(df, output_dir, task_name, model)

    return results_df


def plot_mean_irae_detection(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    # Reshape the data for plotting
    irae_detection_summary_melted = df.melt(
        id_vars=["type", "temperature"],
        value_vars=["average_score"],
        var_name="metric",
        value_name="score",
    )

    # Plotting
    temperatures = sorted(irae_detection_summary_melted["temperature"].unique())
    string_types = irae_detection_summary_melted["type"].unique()

    x = np.arange(len(temperatures))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, string_type in enumerate(string_types):
        subset = irae_detection_summary_melted[
            irae_detection_summary_melted["type"] == string_type
        ]
        ax.bar(x + i * width, subset["score"], width, label=string_type)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean iRAE Detection Score")
    ax.set_title(f"Mean {task_name} by Temperature for {model_name}")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(temperatures)
    ax.legend(title="String Type")
    ax.set_ylim(0, 5)  # Adjust this based on your actual score range

    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, f"{task_name}_{model_name}_mean_plot.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to {plot_file}")


def plot_irae_detection_counts(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    # Ensure the responses are integers
    for temp in ["0.0", "0.7", "1.0"]:
        df[f"response_{temp}"] = df[f"response_{temp}"].apply(
            lambda x: int(x) if str(x).isdigit() else 0
        )

    # Calculate iRAE detection counts for brand and generic drugs
    def calculate_counts(data):
        counts = {}
        for temp in ["0.0", "0.7", "1.0"]:
            counts[temp] = data[f"response_{temp}"].value_counts().sort_index()
        return counts

    counts_brand = calculate_counts(df[df["type"] == "brand"])
    counts_generic = calculate_counts(df[df["type"] == "generic"])

    # Prepare data for plotting
    temperatures = ["0.0", "0.7", "1.0"]
    irae_levels = [0, 1, 2, 3, 4]
    irae_colors = ["gray", "green", "yellow", "orange", "red"]
    irae_labels = {
        0: "NA",
        1: "Very Unlikely",
        2: "Unlikely",
        3: "Likely",
        4: "Very Likely",
    }

    x = np.arange(len(temperatures) * 2)  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot bars for each iRAE level
    for i, level in enumerate(irae_levels):
        brand_counts = [counts_brand[temp].get(level, 0) for temp in temperatures]
        generic_counts = [counts_generic[temp].get(level, 0) for temp in temperatures]

        ax.bar(
            x[::2] - width * 2 + i * width,
            brand_counts,
            width,
            label=f"{level} ({irae_labels[level]})",
            color=irae_colors[i],
        )
        ax.bar(
            x[1::2] - width * 2 + i * width, generic_counts, width, color=irae_colors[i]
        )

    # Customize the plot
    ax.set_xlabel("Temperature and Drug Type")
    ax.set_ylabel("Count")
    ax.set_title(f"iRAE Detection Response Distribution - {model_name}")
    ax.set_xticks(x - width)
    ax.set_xticklabels(
        [
            f'{temp}\n{"Brand" if i%2==0 else "Generic"}'
            for temp in temperatures
            for i in range(2)
        ]
    )
    ax.legend(title="iRAE Level", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add vertical lines to separate temperature groups
    for i in range(1, len(temperatures)):
        ax.axvline(x=i * 2 - 0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, f"{task_name}_{model_name}_counts_plot.png")
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to {plot_file}")


def load_drug_info(brand_file_path, generic_file_path):
    brand_df = pd.read_csv(brand_file_path)
    generic_df = pd.read_csv(generic_file_path)
    combined_df = pd.concat([brand_df, generic_df], ignore_index=True)
    return combined_df[["unique_id", "drug"]].set_index("unique_id").to_dict()["drug"]


def process_differential(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    # Load drug information
    drug_info = load_drug_info(
        "src/irAE/generated_prompts_brand_only.csv",
        "src/irAE/generated_prompts_generic_only.csv",
    )

    def extract_list(text):
        if pd.isna(text):
            return []
        text = re.sub(r"```python|```", "", text)
        try:
            return ast.literal_eval(text)
        except:
            return [item.strip() for item in text.split("\n") if item.strip()]

    def count_events(diagnosis_list, drug):
        if pd.isna(drug) or not isinstance(drug, str):
            drug = ""
        general_pattern = (
            r"adverse\s+event|side\s+effect|complication|toxicit(?:y|ies)|induced"
        )
        irae_pattern = r"immune[\s-]*related|irAE"
        drug_pattern = re.escape(drug) if drug else r"$^"

        general_count = sum(
            1
            for item in diagnosis_list
            if re.search(general_pattern, item, re.IGNORECASE)
        )
        irae_count = sum(
            1 for item in diagnosis_list if re.search(irae_pattern, item, re.IGNORECASE)
        )
        drug_count = sum(
            1 for item in diagnosis_list if re.search(drug_pattern, item, re.IGNORECASE)
        )

        return general_count, irae_count, drug_count

    def count_irae_positions(diagnosis_list):
        irae_pattern = r"immune[\s-]*related|irAE"
        positions = [
            i + 1
            for i, item in enumerate(diagnosis_list)
            if re.search(irae_pattern, item, re.IGNORECASE)
        ]
        return positions

    # Process each temperature
    for temp in ["0.0", "0.7", "1.0"]:
        df[f"list_{temp}"] = df[f"response_{temp}"].apply(extract_list)
        df["drug"] = df["unique_id"].map(drug_info).fillna("")
        df[[f"general_count_{temp}", f"irae_count_{temp}", f"drug_count_{temp}"]] = (
            df.apply(
                lambda row: pd.Series(count_events(row[f"list_{temp}"], row["drug"])),
                axis=1,
            )
        )
        df[f"total_items_{temp}"] = df[f"list_{temp}"].apply(len)
        df[f"irae_positions_{temp}"] = df[f"list_{temp}"].apply(count_irae_positions)

    # Calculate percentages
    for temp in ["0.0", "0.7", "1.0"]:
        for event_type in ["general", "irae", "drug"]:
            df[f"{event_type}_percentage_{temp}"] = (
                df[f"{event_type}_count_{temp}"] / df[f"total_items_{temp}"]
            ) * 100

    # Group by type and calculate mean percentages
    summary = (
        df.groupby("type")
        .agg(
            {
                "general_percentage_0.0": "mean",
                "general_percentage_0.7": "mean",
                "general_percentage_1.0": "mean",
                "irae_percentage_0.0": "mean",
                "irae_percentage_0.7": "mean",
                "irae_percentage_1.0": "mean",
                "drug_percentage_0.0": "mean",
                "drug_percentage_0.7": "mean",
                "drug_percentage_1.0": "mean",
            }
        )
        .reset_index()
    )

    # Plot results
    plot_differential_results(summary, output_dir, task_name, model_name)
    plot_irae_positions_brand_vs_generic(df, output_dir, task_name, model_name)

    return df


def plot_differential_results(
    summary: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    temperatures = ["0.0", "0.7", "1.0"]
    event_types = ["general", "irae", "drug"]
    colors = ["blue", "red", "green"]

    fig, ax = plt.subplots(figsize=(15, 8))

    bar_width = 0.25
    index = range(len(summary["type"]))

    for i, event_type in enumerate(event_types):
        for j, temp in enumerate(temperatures):
            ax.bar(
                [x + (i * 3 + j) * bar_width for x in index],
                summary[f"{event_type}_percentage_{temp}"],
                bar_width,
                label=f"{event_type.capitalize()} (Temp {temp})",
                color=colors[i],
                alpha=0.5 + 0.25 * j,
            )

    ax.set_xlabel("Drug Type")
    ax.set_ylabel("Percentage of Mentions")
    ax.set_title(f"Event Mentions in Differential Diagnosis - {model_name}")
    ax.set_xticks([x + 4 * bar_width for x in index])
    ax.set_xticklabels(summary["type"])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    plot_file = os.path.join(
        output_dir, f"{task_name}_{model_name}_differential_plot.png"
    )
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to {plot_file}")


def plot_irae_positions_brand_vs_generic(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    temperatures = ["0.0", "0.7", "1.0"]
    drug_types = ["brand", "generic"]
    colors = {"brand": "blue", "generic": "red"}

    for temp in temperatures:
        fig, ax = plt.subplots(figsize=(12, 6))

        for drug_type in drug_types:
            df_subset = df[df["type"] == drug_type]
            positions = [
                pos
                for sublist in df_subset[f"irae_positions_{temp}"]
                for pos in sublist
            ]

            ax.hist(
                positions,
                bins=range(1, max(max(positions), 10) + 2),  # Ensure at least 10 bins
                alpha=0.5,
                label=f"{drug_type.capitalize()} Drugs",
                color=colors[drug_type],
            )

        ax.set_xlabel("Position in Differential Diagnosis")
        ax.set_ylabel("Number of irAE Mentions")
        ax.set_title(f"irAE Mention Positions - Temperature {temp} - {model_name}")
        ax.legend()

        plt.tight_layout()

        plot_file = os.path.join(
            output_dir, f"{task_name}_{model_name}_irae_positions_temp_{temp}_plot.png"
        )
        plt.savefig(plot_file)
        plt.close()

        print(f"irAE positions plot for temperature {temp} saved to {plot_file}")
