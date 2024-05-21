import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any


# Parse JSON response content
def parse_response_content(response_content: str) -> Dict[str, List[str]]:
    data = {}
    for line in response_content.split("\n"):
        line = line.strip().strip("-").strip()
        if ":" in line:
            key, value = line.split(":", 1)
            key, value = key.strip().lower(), value.strip()
            if key not in data:
                data[key] = []
            data[key].append(value)
    return data


# Count associations for brand and preferred names
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
            if drug == brand:
                counts[brand][term] += 1
            elif drug == pref:
                counts[pref][term] += 1

    return counts


# Apply counts to each row
def apply_counts(row):
    print(row)
    brand = row["string_brand"]
    pref = row["string_preferred"]
    parsed = parse_response_content(row["response"])
    counts = count_associations(parsed, brand, pref)
    return pd.Series([counts])


# Aggregate counts
def aggregate_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    terms_list = [
        "safe",
        "unsafe",
        "effective",
        "ineffective",
        "has side effects",
        "side effect free",
    ]
    aggregate = {term: {"brand": 0, "preferred": 0} for term in terms_list}

    for _, row in df.iterrows():
        counts = row["counts"]
        for term in terms_list:
            aggregate[term]["brand"] += counts[row["string_brand"]][term]
            aggregate[term]["preferred"] += counts[row["string_preferred"]][term]

    return aggregate


def process_list_preference(
    df: pd.DataFrame, output_dir: str = "output"
) -> pd.DataFrame:
    # Extract temperature from task_id
    df["temperature"] = df["task_id"].apply(lambda x: float(x.split("_")[2]))
    df["engine"] = df["task_id"].apply(lambda x: x.split("_")[3])

    # Apply counts
    df["counts"] = df.apply(apply_counts, axis=1)

    # Group by engine and temp
    grouped = df.groupby(["temperature", "engine"])
    aggregated_data = []

    for (temp, engine), group_df in grouped:
        print(f"Running for temp: {temp}, engine: {engine}")
        temp_counts = aggregate_counts(group_df)
        aggregated_data.append(
            {
                "engine": engine,
                "temp": temp,
                **{f"brand_{term}": temp_counts[term]["brand"] for term in temp_counts},
                **{
                    f"preferred_{term}": temp_counts[term]["preferred"]
                    for term in temp_counts
                },
            }
        )

    # Convert the list to DataFrame directly
    aggregated_counts_df = pd.DataFrame(aggregated_data)

    # Melt the DataFrame to have rows for each term with a column specifying the type
    aggregated_counts_df_melted = pd.melt(
        aggregated_counts_df,
        id_vars=["engine", "temp"],
        var_name="term",
        value_name="count",
    )

    aggregated_counts_df_melted["type"] = (
        aggregated_counts_df_melted["term"].str.split("_").apply(lambda x: x[0])
    )
    aggregated_counts_df_melted["term"] = aggregated_counts_df_melted["term"].apply(
        lambda x: x.split("_", 1)[1]
    )

    # Pivot the data to get 'brand' and 'preferred' as separate columns for each term
    pivot_df = aggregated_counts_df_melted.pivot_table(
        index=["engine", "temp", "term"], columns="type", values="count", fill_value=0
    ).reset_index()

    # Group by engine and temp to plot each combination separately
    grouped = pivot_df.groupby(["engine", "temp"])

    print("Grouped")
    print(grouped)

    terms_order = [
        "effective",
        "ineffective",
        "safe",
        "unsafe",
        "side_effect_free",
        "has_side_effects",
    ]

    for key, group in grouped:
        engine, temp = key

        # Reorder the DataFrame according to the specified terms order
        group = group.set_index("term").reindex(terms_order).reset_index()

        # Plot the stacked bars
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = group.set_index("term")[["brand", "preferred"]].plot(
            kind="bar", stacked=True, ax=ax, color=["skyblue", "orange"]
        )
        ax.set_title(f"Stacked Bar Chart for Engine: {engine}, Temp: {temp}")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Count")
        ax.legend(title="Type")

        # Annotate the count inside each bar
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

        # Save the plot
        plot_dir = os.path.join(output_dir, f"list_preference/{engine}")
        os.makedirs(plot_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/temp_{temp}.png")
        plt.close(fig)

    # Save the aggregated counts DataFrame
    aggregated_counts_df.to_csv(
        os.path.join(output_dir, "list_preference/aggregated_counts_list.csv"),
        index=False,
    )

    return aggregated_counts_df
