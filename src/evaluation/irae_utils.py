import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import ast
from matplotlib.colors import LinearSegmentedColormap
import csv
from scipy.stats import ttest_ind

import pandas as pd
import numpy as np
from scipy import stats


def process_irae_detection(df, output_dir, task_name, model):
    # Process irae detection data
    # Convert string responses to numeric values
    for temp in ["0.0", "0.7", "1.0"]:
        df[f"score_{temp}"] = pd.to_numeric(df[f"response_{temp}"], errors="coerce")

    # Calculate statistics
    results = []
    stats_results = []
    for temp in ["0.0", "0.7", "1.0"]:
        temp_stats = (
            df.groupby("type")[f"score_{temp}"]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        temp_stats["temperature"] = temp
        results.append(temp_stats)

        # Perform statistical test
        brand_data = df[df["type"] == "brand"][f"score_{temp}"]
        generic_data = df[df["type"] == "generic"][f"score_{temp}"]

        t_stat, p_value = stats.ttest_ind(brand_data, generic_data)

        stats_results.append(
            {"temperature": temp, "t_statistic": t_stat, "p_value": p_value}
        )

    results_df = pd.concat(results, ignore_index=True)
    results_df = results_df.rename(
        columns={"mean": "average_score", "median": "median_score", "std": "std_dev"}
    )

    stats_df = pd.DataFrame(stats_results)

    # save results to csv
    results_df.to_csv(f"{output_dir}/{task_name}_{model}_results.csv", index=False)
    print(
        f"Saved results for {model} {task_name} to {output_dir}/{task_name}_{model}_results.csv"
    )

    # save statistical test results to csv
    stats_df.to_csv(f"{output_dir}/{task_name}_{model}_stats.csv", index=False)
    print(
        f"Saved statistical test results for {model} {task_name} to {output_dir}/{task_name}_{model}_stats.csv"
    )

    print("")
    # Plot the results
    plot_mean_irae_detection(results_df, output_dir, task_name, model)

    print(df.columns)
    # Plot the count distribution
    plot_irae_detection_counts(df, output_dir, task_name, model)

    # plot temperature 0.0
    plot_irae_detection_temp_0(df, output_dir, task_name, model)

    return results_df


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
        df["drug"] = df["unique_id"].apply(extract_drug_from_id)
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
                f"{event_type}_percentage_{temp}": ["mean", "std"]
                for event_type in ["general", "irae", "drug"]
                for temp in ["0.0", "0.7", "1.0"]
            }
        )
        .reset_index()
    )

    # Flatten column names
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns
    ]

    # Perform statistical tests
    stats_results = []
    for temp in ["0.0", "0.7", "1.0"]:
        for event_type in ["general", "irae", "drug"]:
            brand_data = df[df["type"] == "brand"][f"{event_type}_percentage_{temp}"]
            generic_data = df[df["type"] == "generic"][
                f"{event_type}_percentage_{temp}"
            ]

            t_stat, p_value = stats.ttest_ind(brand_data, generic_data)

            stats_results.append(
                {
                    "temperature": temp,
                    "event_type": event_type,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                }
            )

    stats_df = pd.DataFrame(stats_results)

    # Save summary to csv
    summary.to_csv(f"{output_dir}/{task_name}_{model_name}_summary.csv", index=False)
    print(
        f"Saved summary for {model_name} {task_name} to {output_dir}/{task_name}_{model_name}_summary.csv"
    )

    # Save statistical test results to csv
    stats_df.to_csv(f"{output_dir}/{task_name}_{model_name}_stats.csv", index=False)
    print(
        f"Saved statistical test results for {model_name} {task_name} to {output_dir}/{task_name}_{model_name}_stats.csv"
    )

    return summary


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

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    colors = {"brand": "#4a7ba7", "generic": "#a77b4a"}  # Muted blue and orange

    for i, string_type in enumerate(string_types):
        subset = irae_detection_summary_melted[
            irae_detection_summary_melted["type"] == string_type
        ]
        bars = ax.bar(
            x + i * width,
            subset["score"],
            width,
            label=string_type,
            color=colors[string_type],
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
    ax.set_ylabel("Mean iRAE Detection Score", fontweight="bold", fontsize=12)
    ax.set_title(
        f"Mean {task_name} by Temperature for {model_name}",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(temperatures, fontweight="bold")
    ax.legend(title="Drug Type", title_fontsize=12, fontsize=10, frameon=False)
    ax.set_ylim(0, 4)  # Adjust this based on your actual score range

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"{task_name}_{model_name}_mean_plot.png")
    plt.savefig(plot_file, bbox_inches="tight")
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
    irae_colors = [
        "#e6e6e6",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#3182bd",
    ]  # Blue color scheme
    irae_labels = {
        0: "NA",
        1: "Very Unlikely",
        2: "Unlikely",
        3: "Likely",
        4: "Very Likely",
    }

    x = np.arange(len(temperatures) * 2)  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

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
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x[1::2] - width * 2 + i * width,
            generic_counts,
            width,
            color=irae_colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    # Customize the plot
    ax.set_xlabel("Temperature and Drug Type", fontweight="bold", fontsize=12)
    ax.set_ylabel("Count", fontweight="bold", fontsize=12)
    ax.set_title(
        f"iRAE Detection Response Distribution - {model_name}",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xticks(x - width)
    ax.set_xticklabels(
        [
            f'{temp}\n{"Brand" if i%2==0 else "Generic"}'
            for temp in temperatures
            for i in range(2)
        ],
        fontweight="bold",
    )
    ax.legend(
        title="iRAE Level",
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Add vertical lines to separate temperature groups
    for i in range(1, len(temperatures)):
        ax.axvline(x=i * 2 - 0.5, color="gray", linestyle="--", alpha=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, f"{task_name}_{model_name}_counts_plot.png")
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {plot_file}")


def plot_irae_detection_temp_0(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    # Ensure the responses are integers
    df["response_0.0"] = df["response_0.0"].apply(
        lambda x: int(x) if str(x).isdigit() else 0
    )

    # Calculate iRAE detection counts for brand and generic drugs
    counts_brand = df[df["type"] == "brand"]["response_0.0"].value_counts().sort_index()
    counts_generic = (
        df[df["type"] == "generic"]["response_0.0"].value_counts().sort_index()
    )

    # Prepare data for plotting
    irae_levels = [0, 1, 2, 3, 4]
    irae_colors = [
        "#e6e6e6",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#3182bd",
    ]  # Blue color scheme
    irae_labels = {
        0: "NA",
        1: "Very Unlikely",
        2: "Unlikely",
        3: "Likely",
        4: "Very Likely",
    }

    x = np.arange(2)  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Plot bars for each iRAE level
    for i, level in enumerate(irae_levels):
        brand_count = counts_brand.get(level, 0)
        generic_count = counts_generic.get(level, 0)

        ax.bar(
            x[0] - width * 2 + i * width,
            brand_count,
            width,
            label=f"{level} ({irae_labels[level]})",
            color=irae_colors[i],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x[1] - width * 2 + i * width,
            generic_count,
            width,
            color=irae_colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    # Customize the plot
    ax.set_xlabel("Drug Type", fontweight="bold", fontsize=12)
    ax.set_ylabel("Count", fontweight="bold", fontsize=12)
    ax.set_title(
        f"iRAE Detection Response Distribution (Temp 0.0) - {model_name}",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xticks(x - width)
    ax.set_xticklabels(["Brand", "Generic"], fontweight="bold")
    ax.legend(
        title="iRAE Level",
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(
        output_dir, f"{task_name}_{model_name}_counts_plot_temp_0.png"
    )
    plt.savefig(plot_file, bbox_inches="tight")
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
        df["drug"] = df["unique_id"].apply(extract_drug_from_id)
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
                f"{event_type}_percentage_{temp}": ["mean", "std"]
                for event_type in ["general", "irae", "drug"]
                for temp in ["0.0", "0.7", "1.0"]
            }
        )
        .reset_index()
    )

    # Flatten column names
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns
    ]

    # Perform statistical tests
    stats_results = []
    for temp in ["0.0", "0.7", "1.0"]:
        for event_type in ["general", "irae", "drug"]:
            brand_data = df[df["type"] == "brand"][f"{event_type}_percentage_{temp}"]
            generic_data = df[df["type"] == "generic"][
                f"{event_type}_percentage_{temp}"
            ]

            t_stat, p_value = ttest_ind(brand_data, generic_data)

            stats_results.append(
                {
                    "temperature": temp,
                    "event_type": event_type,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                }
            )

    stats_df = pd.DataFrame(stats_results)

    # Save summary to csv
    summary.to_csv(f"{output_dir}/{task_name}_{model_name}_summary.csv", index=False)
    print(
        f"Saved summary for {model_name} {task_name} to {output_dir}/{task_name}_{model_name}_summary.csv"
    )

    # Save statistical test results to csv
    stats_df.to_csv(f"{output_dir}/{task_name}_{model_name}_stats.csv", index=False)
    print(
        f"Saved statistical test results for {model_name} {task_name} to {output_dir}/{task_name}_{model_name}_stats.csv"
    )

    return summary


def plot_differential_results(
    summary: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    temperatures = ["0.0", "0.7", "1.0"]
    event_types = ["general", "irae", "drug"]
    colors = {"brand": "#4a7ba7", "generic": "#a77b4a"}  # Muted blue  # Muted orange

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
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
            edgecolor="black",
            linewidth=0.5,
        )
        rects2 = ax.bar(
            x + width / 2,
            generic_data,
            width,
            label="Generic",
            color=colors["generic"],
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_ylabel("Percentage of Mentions", fontweight="bold")
        ax.set_title(f"Temperature {temp}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(event_types, fontweight="bold")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(0, 100)

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
                    fontweight="bold",
                )

        autolabel(rects1)
        autolabel(rects2)

    plt.tight_layout()
    plot_file = os.path.join(
        output_dir, f"{task_name}_{model_name}_differential_plot.png"
    )
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {plot_file}")


# def plot_irae_positions_brand_vs_generic(
#     df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
# ):
#     temperatures = ["0.0", "0.7", "1.0"]
#     drug_types = ["brand", "generic"]
#     event_types = ["drug", "general", "irae"]
#     colors = {
#         "drug": "#4a7ba7",  # Muted blue
#         "general": "#a77b4a",  # Muted orange
#         "irae": "#4aa77b",  # Muted green
#     }

#     for temp in temperatures:
#         fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
#         fig.suptitle(
#             f"Event Mention Positions - Temperature {temp} - {model_name}",
#             fontsize=16,
#             fontweight="bold",
#         )

#         for i, drug_type in enumerate(drug_types):
#             df_subset = df[df["type"] == drug_type]

#             for j, event_type in enumerate(event_types):
#                 positions = []
#                 for _, row in df_subset.iterrows():
#                     pattern = ""
#                     if event_type == "drug":
#                         pattern = (
#                             r"\b"
#                             + re.escape(row["drug"])
#                             + r"\b|\b"
#                             + r"\b|\b".join(row["drug"].split())
#                             + r"\b"
#                         )
#                     elif event_type == "general":
#                         pattern = r"adverse\s+event|side\s+effect|complication|toxicit(?:y|ies)|induced"
#                     elif event_type == "irae":
#                         pattern = r"immune[\s-]*related|irAE"

#                     event_positions = [
#                         idx + 1
#                         for idx, item in enumerate(row[f"list_{temp}"])
#                         if re.search(pattern, item, re.IGNORECASE)
#                     ]
#                     positions.extend(event_positions)

#                 counts, bin_edges = np.histogram(positions, bins=range(1, 16))
#                 total_mentions = sum(counts)

#                 # Create custom colormap
#                 n_bins = len(counts)
#                 cmap = LinearSegmentedColormap.from_list(
#                     "custom", [colors[event_type], "white"]
#                 )
#                 colors_list = [cmap(i / n_bins) for i in range(n_bins)]

#                 bars = axes[i, j].bar(
#                     bin_edges[:-1],
#                     counts,
#                     align="edge",
#                     width=0.8,
#                     color=colors_list,
#                     edgecolor="black",
#                     linewidth=0.5,
#                 )

#                 for bar in bars:
#                     height = bar.get_height()
#                     if total_mentions > 0:
#                         percentage = (height / total_mentions) * 100
#                         axes[i, j].text(
#                             bar.get_x() + bar.get_width() / 2,
#                             height,
#                             f"{percentage:.1f}%",
#                             ha="center",
#                             va="bottom",
#                             fontsize=8,
#                             fontweight="bold",
#                             rotation=90,
#                         )

#                 axes[i, j].set_xlabel(
#                     "Position in Differential Diagnosis", fontweight="bold"
#                 )
#                 axes[i, j].set_ylabel("Number of Mentions", fontweight="bold")
#                 axes[i, j].set_title(
#                     f"{drug_type.capitalize()} - {event_type.capitalize()}",
#                     fontweight="bold",
#                 )
#                 axes[i, j].grid(axis="y", linestyle="--", alpha=0.3)
#                 axes[i, j].spines["top"].set_visible(False)
#                 axes[i, j].spines["right"].set_visible(False)

#                 axes[i, j].text(
#                     0.95,
#                     0.95,
#                     f"Total: {total_mentions}",
#                     transform=axes[i, j].transAxes,
#                     verticalalignment="top",
#                     horizontalalignment="right",
#                     fontweight="bold",
#                 )

#         plt.tight_layout()
#         plot_file = os.path.join(
#             output_dir, f"{task_name}_{model_name}_event_positions_temp_{temp}_plot.png"
#         )
#         plt.savefig(plot_file, bbox_inches="tight")
#         plt.close()

#         print(f"Event positions plot for temperature {temp} saved to {plot_file}")


def plot_irae_positions_brand_vs_generic(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    temperatures = ["0.0", "0.7", "1.0"]
    event_types = ["drug", "general", "irae"]
    colors = {
        "brand": {
            "drug": "#4a7ba7",  # Muted blue
            "general": "#a77b4a",  # Muted orange
            "irae": "#4aa77b",  # Muted green
        },
        "generic": {
            "drug": "#7ba74a",  # Muted green-blue
            "general": "#a74a7b",  # Muted purple
            "irae": "#7b4aa7",  # Muted blue-purple
        },
    }

    for temp in temperatures:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
        fig.suptitle(
            f"Event Mention Positions - Temperature {temp} - {model_name}",
            fontsize=16,
            fontweight="bold",
        )

        for j, event_type in enumerate(event_types):
            brand_positions = []
            generic_positions = []

            for _, row in df.iterrows():
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
                    min(idx + 1, 3)  # Limit positions to 1, 2, or 3
                    for idx, item in enumerate(row[f"list_{temp}"])
                    if re.search(pattern, item, re.IGNORECASE)
                ]

                if row["type"] == "brand":
                    brand_positions.extend(event_positions)
                else:
                    generic_positions.extend(event_positions)

            brand_counts = np.histogram(brand_positions, bins=[1, 2, 3, 4])[0]
            generic_counts = np.histogram(generic_positions, bins=[1, 2, 3, 4])[0]

            x = np.arange(3)
            width = 0.35

            axes[j].bar(
                x - width / 2,
                brand_counts,
                width,
                label="Brand",
                color=colors["brand"][event_type],
                edgecolor="black",
                linewidth=0.5,
            )
            axes[j].bar(
                x + width / 2,
                generic_counts,
                width,
                label="Generic",
                color=colors["generic"][event_type],
                edgecolor="black",
                linewidth=0.5,
            )

            axes[j].set_xlabel("Position in Differential Diagnosis", fontweight="bold")
            axes[j].set_ylabel("Number of Mentions", fontweight="bold")
            axes[j].set_title(f"{event_type.capitalize()}", fontweight="bold")
            axes[j].set_xticks(x)
            axes[j].set_xticklabels(["1", "2", "3"])
            axes[j].legend()
            axes[j].grid(axis="y", linestyle="--", alpha=0.3)
            axes[j].spines["top"].set_visible(False)
            axes[j].spines["right"].set_visible(False)

            for i, count in enumerate(brand_counts):
                axes[j].text(i - width / 2, count, str(count), ha="center", va="bottom")
            for i, count in enumerate(generic_counts):
                axes[j].text(i + width / 2, count, str(count), ha="center", va="bottom")

            total_mentions = sum(brand_counts) + sum(generic_counts)
            axes[j].text(
                0.95,
                0.95,
                f"Total: {total_mentions}",
                transform=axes[j].transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontweight="bold",
            )

        plt.tight_layout()
        plot_file = os.path.join(
            output_dir, f"{task_name}_{model_name}_event_positions_temp_{temp}_plot.png"
        )
        plt.savefig(plot_file, bbox_inches="tight")
        plt.close()

        print(f"Event positions plot for temperature {temp} saved to {plot_file}")
