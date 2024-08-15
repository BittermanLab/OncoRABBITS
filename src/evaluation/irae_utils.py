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

    # save to csv
    results_df.to_csv(f"{output_dir}/{task_name}_{model}_results.csv", index=False)
    print(
        f"Saved results for {model} {task_name} to {output_dir}/{task_name}_{model}_results.csv"
    )
    print("")
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

    def extract_drug_from_id(unique_id):
        parts = unique_id.split("_")
        return parts[1] if len(parts) > 1 else ""

    def count_events(diagnosis_list, drug):
        if pd.isna(drug) or not isinstance(drug, str):
            drug = ""
        general_pattern = (
            r"adverse\s+event|side\s+effect|complication|toxicit(?:y|ies)|induced"
        )
        irae_pattern = r"immune[\s-]*related|irAE"
        drug_pattern = (
            r"\b" + re.escape(drug) + r"\b|\b" + r"\b|\b".join(drug.split()) + r"\b"
            if drug
            else r"$^"
        )

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
        df["drug"] = df["unique_id"].apply(extract_drug_from_id)  # Changed this line
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
                f"{event_type}_percentage_{temp}": "mean"
                for event_type in ["general", "irae", "drug"]
                for temp in ["0.0", "0.7", "1.0"]
            }
        )
        .reset_index()
    )

    print("Debug - Summary:")

    # save to csv
    summary.to_csv(f"{output_dir}/{task_name}_{model_name}_summary.csv", index=False)
    print(
        f"Saved summary for {model_name} {task_name} to {output_dir}/{task_name}_{model_name}_summary.csv"
    )
    print("")
    # Plot results
    plot_differential_results(summary, output_dir, task_name, model_name)
    plot_irae_positions_brand_vs_generic(df, output_dir, task_name, model_name)

    return df


def plot_differential_results(
    summary: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    temperatures = ["0.0", "0.7", "1.0"]
    event_types = ["general", "irae", "drug"]
    colors = {
        "brand": "#4E79A7",
        "generic": "#F28E2B",
    }  # Blue for brand, Orange for generic

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Event Mentions in Differential Diagnosis - {model_name}",
        fontweight="bold",
        fontsize=16,
    )

    for i, temp in enumerate(temperatures):
        ax = axes[i]

        x = np.arange(len(event_types))
        width = 0.35

        brand_data = summary[summary["type"] == "brand"][
            [f"{et}_percentage_{temp}" for et in event_types]
        ].values[0]
        generic_data = summary[summary["type"] == "generic"][
            [f"{et}_percentage_{temp}" for et in event_types]
        ].values[0]

        rects1 = ax.bar(
            x - width / 2,
            brand_data,
            width,
            label="Brand",
            color=colors["brand"],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        rects2 = ax.bar(
            x + width / 2,
            generic_data,
            width,
            label="Generic",
            color=colors["generic"],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_ylabel("Percentage of Mentions", fontweight="bold")
        ax.set_title(f"Temperature {temp}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(event_types, fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Add value labels on the bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

        autolabel(rects1)
        autolabel(rects2)

    plt.tight_layout()

    plot_file = os.path.join(
        output_dir, f"{task_name}_{model_name}_differential_plot.png"
    )
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {plot_file}")


def plot_irae_positions_brand_vs_generic(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    temperatures = ["0.0", "0.7", "1.0"]
    drug_types = ["brand", "generic"]
    event_types = ["drug", "general", "irae"]
    colors = {
        "drug": "#4E79A7",  # Blue
        "general": "#F28E2B",  # Orange
        "irae": "#59A14F",  # Green
    }

    for temp in temperatures:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f"Event Mention Positions - Temperature {temp} - {model_name}", fontsize=16
        )

        for i, drug_type in enumerate(drug_types):
            df_subset = df[df["type"] == drug_type]

            for j, event_type in enumerate(event_types):
                positions = []
                for _, row in df_subset.iterrows():
                    pattern = ""
                    if event_type == "drug":
                        pattern = (
                            r"\b"
                            + re.escape(row["drug"])
                            + r"\b|\b"
                            + r"\b|\b".join(row["drug"].split())
                            + r"\b"
                        )
                    elif event_type == "general":
                        pattern = r"adverse\s+event|side\s+effect|complication|toxicit(?:y|ies)|induced"
                    elif event_type == "irae":
                        pattern = r"immune[\s-]*related|irAE"

                    event_positions = [
                        idx + 1
                        for idx, item in enumerate(row[f"list_{temp}"])
                        if re.search(pattern, item, re.IGNORECASE)
                    ]
                    positions.extend(event_positions)

                # Calculate histogram data
                counts, bin_edges = np.histogram(positions, bins=range(1, 16))
                total_mentions = sum(counts)

                # Plot histogram
                bars = axes[i, j].bar(
                    bin_edges[:-1],
                    counts,
                    align="edge",
                    width=0.8,
                    alpha=0.7,
                    color=colors[event_type],
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add percentage labels
                for bar in bars:
                    height = bar.get_height()
                    if total_mentions > 0:
                        percentage = (height / total_mentions) * 100
                        axes[i, j].text(
                            bar.get_x() + bar.get_width() / 2,
                            height,
                            f"{percentage:.1f}%",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=90,
                        )

                axes[i, j].set_xlabel(
                    "Position in Differential Diagnosis", fontweight="bold"
                )
                axes[i, j].set_ylabel("Number of Mentions", fontweight="bold")
                axes[i, j].set_title(
                    f"{drug_type.capitalize()} - {event_type.capitalize()}",
                    fontweight="bold",
                )
                axes[i, j].grid(axis="y", linestyle="--", alpha=0.7)
                axes[i, j].spines["top"].set_visible(False)
                axes[i, j].spines["right"].set_visible(False)

                # Add count of total mentions
                axes[i, j].text(
                    0.95,
                    0.95,
                    f"Total: {total_mentions}",
                    transform=axes[i, j].transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                )

        plt.tight_layout()

        plot_file = os.path.join(
            output_dir, f"{task_name}_{model_name}_event_positions_temp_{temp}_plot.png"
        )
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Event positions plot for temperature {temp} saved to {plot_file}")
