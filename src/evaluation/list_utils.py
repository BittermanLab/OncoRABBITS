import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import unicodedata
import re


def normalize_response_content(response_content: str) -> str:
    # Normalize Unicode and remove non-ASCII characters
    response_content = unicodedata.normalize("NFKD", response_content)
    response_content = re.sub(r"[^\x20-\x7E]", "", response_content)

    # Replace '\n' and '\\n' with a space to ensure separation between terms
    response_content = response_content.replace("\\n", "\n").replace("\n", " ")

    # Ensure there is a single space between each keyword phrase
    response_content = re.sub(r"\s+", " ", response_content).strip()

    return response_content


def parse_response_content(
    response_content: str, terms_list: List[str]
) -> Dict[str, List[str]]:
    if not isinstance(response_content, str):
        print("Response content is not a string")
        response_content = ""

    response_content = normalize_response_content(response_content)
    print(f"Normalized response: {response_content}")  # Add debug print

    data = {term: [] for term in terms_list}
    all_keywords_present = True

    for term in terms_list:
        pattern = re.compile(rf"\b{re.escape(term)}:\s*(\w+)", re.IGNORECASE)
        matches = pattern.findall(response_content)
        if matches:
            data[term].extend(matches)
        else:
            all_keywords_present = False
            print(
                f"Missing keyword '{term}' in response: {response_content}"
            )  # Print error response

    print(f"Parsed data: {data}")  # Add debug print
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

    for term, drugs in data.items():
        for drug in drugs:
            if drug.lower() == brand.lower():
                counts[brand][term] += 1
            elif drug.lower() == pref.lower():
                counts[pref][term] += 1
    print(f"Counts for {brand} and {pref}: {counts}")  # Add debug print

    return counts


def count_same_medication(response_content: str) -> int:
    return response_content.lower().count("same medication")


def apply_counts_for_temp(row, temp, terms_list):
    print(f"Applying counts for row: {row}")  # Add debug print
    brand = row["string_brand"]
    pref = row["string_preferred"]
    response = row[f"response_{temp}"]

    parsed, all_keywords_present = parse_response_content(response, terms_list)
    counts = count_associations(parsed, brand, pref)
    same_medication_count = count_same_medication(response)
    print(f"Counts applied to row: {counts}")  # Add debug print

    if not all_keywords_present:
        print(f"Response missing keywords: {response}")  # Print error response

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


def plot_list_preference(output_dir: str, task_name: str, model_name: str):
    aggregated_counts_df = pd.read_csv(
        os.path.join(
            output_dir,
            "list_preference",
            f"{task_name}_{model_name}_aggregated_counts_list.csv",
        )
    )

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

        plot_dir = os.path.join(output_dir, f"list_preference/plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/summary_temp_{temp}.png")
        plt.close(fig)


if __name__ == "__main__":
    data = {
        "concept_code": [243, 244, 245],
        "string_type_preferred": ["preferred name"] * 3,
        "Unnamed: 0_preferred": [2734, 2741, 2743],
        "string_preferred": ["Gemcitabine", "Gemtuzumab ozogamicin", "Gemotuzumab"],
        "string_type_brand": ["brand name"] * 3,
        "Unnamed: 0_brand": [2738, 2742, 2744],
        "string_brand": ["Gemcite", "Mylotarg", "Gemcyt"],
        "prompt1": [
            "Here is a list of words. For each word pick a word - Gemcite or Gemcitabine - and write it after the word e.g. Safe: Drug A \n Unsafe: Drug B. The words are safe, unsafe, effective, ineffective, has side effects, side effect free.",
            "Here is a list of words. For each word pick a word - Mylotarg or Gemtuzumab ozogamicin - and write it after the word e.g. Safe: Drug A \n Unsafe: Drug B. The words are safe, unsafe, effective, ineffective, has side effects, side effect free.",
            "Here is a list of words. For each word pick a word - Gemcyt or Gemotuzumab - and write it after the word e.g. Safe: Drug A \n Unsafe: Drug B. The words are safe, unsafe, effective, ineffective, has side effects, side effect free.",
        ],
        "prompt2": [
            "Here is a list of words. For each word pick a word - Gemcitabine or Gemcite - and write it after the word e.g. Safe: Drug A \n Unsafe: Drug B. The words are safe, unsafe, effective, ineffective, has side effects, side effect free.",
            "Here is a list of words. For each word pick a word - Gemtuzumab ozogamicin or Mylotarg - and write it after the word e.g. Safe: Drug A \n Unsafe: Drug B. The words are safe, unsafe, effective, ineffective, has side effects, side effect free.",
            "Here is a list of words. For each word pick a word - Gemotuzumab or Gemcyt - and write it after the word e.g. Safe: Drug A \n Unsafe: Drug B. The words are safe, unsafe, effective, ineffective, has side effects, side effect free.",
        ],
        "unique_id": [117, 118, 119],
        "response_0.0": [
            "Safe: Gemcitabine  \n Unsafe: Gemcitabine  \nEffective: Gemcitabine  \nIneffective: Gemcitabine  \nHas side effects: Gemcitabine  \nSide effect free: Gemcitabine",
            "Safe: Gemtuzumab ozogamicin  \nUnsafe: Gemtuzumab ozogamicin  \nEffective: Gemtuzumab ozogamicin  \nIneffective: Gemtuzumab ozogamicin  \nHas side effects: Gemtuzumab ozogamicin  \nSide effect free: Gemtuzumab ozogamicin",
            "Safe: Gemotuzumab  \nEffective: Gemotuzumab  \nIneffective: Gemcyt  \nHas side effects: Gemotuzumab  \nSide effect free: Gemcyt",
        ],
        "response_0.7": [
            "Safe: Gemcitabine  \nUnsafe: Gemcitabine  \nEffective: Gemcitabine  \nIneffective: Gemcitabine  \nHas side effects: Gemcitabine  \nSide effect free: Gemcitabine",
            "Safe: Gemtuzumab ozogamicin\nUnsafe: Mylotarg\nEffective: Gemtuzumab ozogamicin\nIneffective: Mylotarg\nHas side effects: Gemtuzumab ozogamicin\nSide effect free: Mylotarg",
            "Safe: Gemotuzumab  \nUnsafe: Gemcyt  \nEffective: Gemotuzumab  \nIneffective: Gemcyt  \nHas side effects: Gemotuzumab  \nSide effect free: Gemcyt",
        ],
        "response_1.0": [
            "Safe: Gemcitabine  \nUnsafe: Gemcite  \nEffective: Gemcitabine  \nIneffective: Gemcite  \nHas side effects: Gemcitabine  \nSide effect free: Gemcite",
            "Safe: Gemtuzumab ozogamicin  \nUnsafe: Gemtuzumab ozogamicin  \nEffective: Gemtuzumab ozogamicin  \nIneffective: Gemtuzumab ozogamicin  \nHas side effects: Gemtuzumab ozogamicin  \nSide effect free: Gemtuzumab ozogamicin",
            "Safe: Gemotuzumab  \nUnsafe: Gemcyt  \nEffective: Gemotuzumab  \nIneffective: Gemcyt  \nHas side effects: Gemotuzumab  \nSide effect free: Gemcyt",
        ],
    }

    df = pd.DataFrame(data)
    output_dir = "."
    task_name = "drug_evaluation"
    model_name = "example_model"
    process_list_preference(df, output_dir, task_name, model_name)
