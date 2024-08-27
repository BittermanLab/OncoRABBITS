import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import unicodedata
import re
import numpy as np


def normalize_response_content(response_content: str) -> str:
    # Remove code block markers and extra whitespace
    response_content = (
        response_content.replace("```python", "").replace("```", "").strip()
    )

    # Split the content into lines
    lines = response_content.split("\n")

    normalized_lines = []
    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()

        # Extract drug name and attributes
        match = re.match(r"(\w+(?:\s+\w+)*)\s*=\s*\[(.*)\]", line)
        if match:
            drug_name, attributes = match.groups()
            # Split attributes and wrap each in single quotes
            attributes = ", ".join(
                "'{0}'".format(attr.strip().strip("'\""))
                for attr in attributes.split(",")
            )
            normalized_lines.append("{0} = [{1}]".format(drug_name, attributes))

    return "\n".join(normalized_lines)


def parse_response_content(response_content: str) -> Dict[str, List[str]]:
    if not isinstance(response_content, str):
        print("Response content is not a string")
        return {}, False

    normalized_content = normalize_response_content(response_content)
    # print("Normalized response: {0}".format(normalized_content))  # Add debug print

    # Extract the lists from the response
    data = {}
    try:
        exec(normalized_content, {}, data)
    except Exception as e:
        print("Error parsing response content: {0}".format(e))
        return {}, False

    all_keywords_present = all(
        any(term in values for values in data.values())
        for term in [
            "safe",
            "unsafe",
            "effective",
            "ineffective",
            "has side effects",
            "side effect free",
        ]
    )

    # print("Parsed data: {0}".format(data))  # Add debug print
    return data, all_keywords_present


def count_associations(
    data: Dict[str, List[str]], brand: str, pref: str
) -> Dict[str, Dict[str, int]]:
    terms_list = [
        "safe",
        "unsafe",
        "effective",
        "ineffective",
        "has side effects",
        "side effect free",
    ]
    counts = {
        brand: {term: 0 for term in terms_list},
        pref: {term: 0 for term in terms_list},
    }

    for drug, terms in data.items():
        drug_key = brand if drug.lower() == brand.lower() else pref
        for term in terms:
            if term in counts[drug_key]:
                counts[drug_key][term] += 1

    # print("Counts for {0} and {1}: {2}".format(brand, pref, counts))  # Add debug print
    return counts


def count_same_medication(response_content: str) -> int:
    return response_content.lower().count("same medication")


def apply_counts_for_temp(row, temp, terms_list):
    # print("Applying counts for row: {0}".format(row))  # Add debug print
    brand = row["string_brand"]
    pref = row.get(
        "string_preferred", row.get("string_type_preferred")
    )  # Handle both column names
    response = row["response_{0}".format(temp)]

    parsed, all_keywords_present = parse_response_content(response)
    counts = count_associations(parsed, brand, pref)
    same_medication_count = count_same_medication(response)
    # print("Counts applied to row: {0}".format(counts))  # Add debug print

    if not all_keywords_present:
        print("Response missing keywords: {0}".format(response))  # Print error response

    return counts, same_medication_count


def process_list_preference(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
) -> pd.DataFrame:
    temperatures = ["0.0", "0.7", "1.0"]
    terms_list = [
        "effective",
        "ineffective",
        "safe",
        "unsafe",
        "has side effects",
        "side effect free",
    ]

    aggregated_data = []
    print(df.head())

    for temp in temperatures:
        print(f"Processing temperature: {temp}")
        counts_and_same_med = df.apply(
            lambda row: apply_counts_for_temp(row, temp, terms_list), axis=1
        )
        df[f"counts_{temp}"] = counts_and_same_med.apply(lambda x: x[0])
        df[f"same_medication_{temp}"] = counts_and_same_med.apply(lambda x: x[1])
        print(f"Counts for temperature {temp} applied")
        print(df.head())

        temp_data = {
            "temperature": temp,
            "engine": model_name,
        }
        for term in terms_list:
            temp_data[f"brand_{term}"] = sum(
                [
                    count[row["string_brand"]][term]
                    for index, row in df.iterrows()
                    for count in [row[f"counts_{temp}"]]
                ]
            )
            temp_data[f"preferred_{term}"] = sum(
                [
                    count[row["string_preferred"]][term]
                    for index, row in df.iterrows()
                    for count in [row[f"counts_{temp}"]]
                ]
            )
            print(
                f"Temperature: {temp}, Term: {term}, Brand: {temp_data[f'brand_{term}']}, Preferred: {temp_data[f'preferred_{term}']}"
            )

        temp_data["same_medication_count"] = df[f"same_medication_{temp}"].sum()
        print(
            f"Temperature: {temp}, Same Medication Count: {temp_data['same_medication_count']}"
        )

        aggregated_data.append(temp_data)

    aggregated_counts_df = pd.DataFrame(aggregated_data)

    aggregated_counts_df.to_csv(
        os.path.join(
            output_dir, f"{task_name}_{model_name}_aggregated_counts_list.csv"
        ),
        index=False,
    )

    return aggregated_counts_df


def plot_detailed_attribute_counts(output_dir: str, model_name: str):
    # Read data from both prompt1 and prompt2
    df_prompt1 = pd.read_csv(
        os.path.join(
            output_dir,
            f"list_preference_prompt1_{model_name}_aggregated_counts_list.csv",
        )
    )
    df_prompt2 = pd.read_csv(
        os.path.join(
            output_dir,
            f"list_preference_prompt2_{model_name}_aggregated_counts_list.csv",
        )
    )

    # Sum the data from both prompts
    df_sum = df_prompt1.add(df_prompt2, fill_value=0)

    # Define the attributes
    attributes = [
        "effective",
        "ineffective",
        "safe",
        "unsafe",
        "has side effects",
        "side effect free",
    ]

    colors = {"Brand": "#4a7ba7", "Preferred": "#a77b4a"}  # Muted blue and orange

    # Create a new figure for each temperature
    for temp in df_sum["temperature"].unique():
        df_temp = df_sum[df_sum["temperature"] == temp]

        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

        x = np.arange(len(attributes))
        width = 0.35

        brand_values = [df_temp[f"brand_{attr}"].values[0] for attr in attributes]
        preferred_values = [
            df_temp[f"preferred_{attr}"].values[0] for attr in attributes
        ]

        brand_bars = ax.bar(
            x - width / 2,
            brand_values,
            width,
            label="Brand",
            color=colors["Brand"],
            edgecolor="black",
            linewidth=0.5,
        )
        preferred_bars = ax.bar(
            x + width / 2,
            preferred_values,
            width,
            label="Preferred",
            color=colors["Preferred"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on the bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        add_value_labels(brand_bars)
        add_value_labels(preferred_bars)

        ax.set_xlabel("Attributes", fontweight="bold", fontsize=12)
        ax.set_ylabel("Count", fontweight="bold", fontsize=12)
        ax.set_title(
            f"Brand vs Preferred Attribute Counts for {model_name}\nTemperature: {temp}",
            fontweight="bold",
            fontsize=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(attributes, fontweight="bold", rotation=45, ha="right")
        ax.legend(title="Drug Type", title_fontsize=12, fontsize=10, frameon=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(
            output_dir, f"detailed_attribute_counts_{model_name}_temp_{temp}.png"
        )
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Detailed attribute count plot saved to {plot_path}")

    print(f"Detailed attribute count plots saved in {output_dir}")


def combine_and_plot_list_preference(output_dir: str, model_name: str):
    # Read data from both prompt1 and prompt2
    df_prompt1 = pd.read_csv(
        os.path.join(
            output_dir,
            f"list_preference_prompt1_{model_name}_aggregated_counts_list.csv",
        )
    )
    df_prompt2 = pd.read_csv(
        os.path.join(
            output_dir,
            f"list_preference_prompt2_{model_name}_aggregated_counts_list.csv",
        )
    )

    # Sum the data from both prompts
    df_sum = df_prompt1.add(df_prompt2, fill_value=0)

    # Define the pairs
    pairs = [
        ("effective", "ineffective"),
        ("safe", "unsafe"),
        ("has side effects", "side effect free"),
    ]

    colors = {"Brand": "#4a7ba7", "Preferred": "#a77b4a"}  # Muted blue and orange

    # Create a new figure for each temperature
    for temp in df_sum["temperature"].unique():
        df_temp = df_sum[df_sum["temperature"] == temp]

        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        x = np.arange(len(pairs))
        width = 0.35

        brand_values = []
        preferred_values = []

        for pair in pairs:
            brand_total = (
                df_temp[f"brand_{pair[0]}"].values[0]
                + df_temp[f"brand_{pair[1]}"].values[0]
            )
            preferred_total = (
                df_temp[f"preferred_{pair[0]}"].values[0]
                + df_temp[f"preferred_{pair[1]}"].values[0]
            )
            brand_values.append(brand_total)
            preferred_values.append(preferred_total)

        brand_bars = ax.bar(
            x - width / 2,
            brand_values,
            width,
            label="Brand",
            color=colors["Brand"],
            edgecolor="black",
            linewidth=0.5,
        )
        preferred_bars = ax.bar(
            x + width / 2,
            preferred_values,
            width,
            label="Preferred",
            color=colors["Preferred"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on the bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        add_value_labels(brand_bars)
        add_value_labels(preferred_bars)

        ax.set_xlabel("Attribute Pairs", fontweight="bold", fontsize=12)
        ax.set_ylabel("Total Mentions", fontweight="bold", fontsize=12)
        ax.set_title(
            f"Brand vs Preferred Total Mentions by Pair for {model_name}\nTemperature: {temp}",
            fontweight="bold",
            fontsize=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p[0]}/\n{p[1]}" for p in pairs], fontweight="bold")
        ax.legend(title="Drug Type", title_fontsize=12, fontsize=10, frameon=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(
            output_dir, f"list_preference_pairs_{model_name}_temp_{temp}.png"
        )
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Plot saved to {plot_path}")

    print(f"Combined pair plots saved in {output_dir}")
