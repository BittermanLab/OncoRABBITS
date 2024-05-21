import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import unicodedata
import re


def parse_response_content(
    response_content: str, terms_list: List[str]
) -> Dict[str, List[str]]:
    if not isinstance(response_content, str):
        print("Response content is not a string")
        response_content = ""

    response_content = unicodedata.normalize("NFKD", response_content)
    response_content = re.sub(r"[^\x20-\x7E]", "", response_content)

    data = {term: [] for term in terms_list}
    lines = response_content.split("\n")
    for line in lines:
        line = line.strip()
        for term in terms_list:
            pattern = re.compile(rf"\b{re.escape(term)}:\s*(\w+)", re.IGNORECASE)
            matches = pattern.findall(line)
            if matches:
                data[term].extend(matches)
    print(f"Parsed data: {data}")  # Add debug print
    return data


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

    for term, drugs in data.items():
        for drug in drugs:
            if drug.lower() == brand.lower():
                counts[brand][term] += 1
            elif drug.lower() == pref.lower():
                counts[pref][term] += 1
    print(f"Counts for {brand} and {pref}: {counts}")  # Add debug print

    return counts


def apply_counts_for_temp(row, temp, terms_list):
    brand = row["string_brand"]
    pref = row["string_preferred"]
    response = row[f"response_{temp}"]

    parsed = parse_response_content(response, terms_list)
    counts = count_associations(parsed, brand, pref)
    print(f"Counts applied to row: {counts}")  # Add debug print
    return counts


def process_list_preference(
    df: pd.DataFrame, output_dir: str, task_name: str
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

    for temp in temperatures:
        print(f"Processing temperature: {temp}")
        df[f"counts_{temp}"] = df.apply(
            lambda row: apply_counts_for_temp(row, temp, terms_list), axis=1
        )
        print(f"Counts for temperature {temp} applied")
        print(df.head())

        temp_data = {
            "temperature": temp,
            "engine": "gpt-4o",
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

        aggregated_data.append(temp_data)

    aggregated_counts_df = pd.DataFrame(aggregated_data)

    aggregated_counts_df_melted = pd.melt(
        aggregated_counts_df,
        id_vars=["engine", "temperature"],
        var_name="term",
        value_name="count",
    )

    aggregated_counts_df_melted["type"] = (
        aggregated_counts_df_melted["term"].str.split("_").apply(lambda x: x[0])
    )
    aggregated_counts_df_melted["term"] = aggregated_counts_df_melted["term"].apply(
        lambda x: "_".join(x.split("_")[1:])
    )

    pivot_df = aggregated_counts_df_melted.pivot_table(
        index=["engine", "temperature", "term"],
        columns="type",
        values="count",
        fill_value=0,
    ).reset_index()

    grouped = pivot_df.groupby(["engine", "temperature"])

    terms_order = [
        "effective",
        "ineffective",
        "safe",
        "unsafe",
        "side effect free",
        "has side effects",
    ]

    for key, group in grouped:
        engine, temp = key

        group = group.set_index("term").reindex(terms_order).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = group.set_index("term")[["brand", "preferred"]].plot(
            kind="bar", stacked=True, ax=ax, color=["skyblue", "orange"]
        )
        ax.set_title(f"Stacked Bar Chart for Engine: {engine}, Temp: {temp}")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Count")
        ax.legend(title="Type")

        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_x(), p.get_y()
            if height > 0:
                ax.text(
                    x + width / 2,
                    y + height / 2,
                    f"{int(height)}",
                    ha="center",
                    va="center",
                )

        plot_dir = os.path.join(output_dir, f"list_preference")
        os.makedirs(plot_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{task_name}_temp_{temp}.png")
        plt.close(fig)

    aggregated_counts_df.to_csv(
        os.path.join(
            output_dir, f"list_preference/{task_name}_aggregated_counts_list.csv"
        ),
        index=False,
    )

    return aggregated_counts_df
