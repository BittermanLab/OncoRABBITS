import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu


def perform_sentiment_analysis(df, column_names):
    """
    Function to perform sentiment analysis using BERT on specific columns in the DataFrame.
    """
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    for column_name in column_names:
        sentiments = []
        for response in df[column_name]:
            try:
                text = response.strip() if isinstance(response, str) else ""

                if text:
                    encoded_input = tokenizer(text, return_tensors="pt")
                    output = model(**encoded_input)
                    scores = output.logits[0].detach().numpy()
                    scores = softmax(scores)
                    sentiment = np.argmax(scores)
                    sentiments.append(sentiment)
                else:
                    sentiments.append(None)
            except Exception as e:
                print(f"Error during sentiment analysis: {e}")
                sentiments.append(None)

        df[f"sentiment_{column_name}"] = sentiments
    return df


def summarize_sentiment(df):
    """
    Function to summarize sentiment values for brand names and generic names,
    comparing them at each temperature.
    """
    if "string_type" not in df:
        df["string_type"] = df["type"]

    temperatures = ["0.0", "0.7", "1.0"]
    results = []

    for temp in temperatures:
        sentiment_col = f"sentiment_response_{temp}"

        # Chi-square test
        contingency_table = pd.crosstab(df["string_type"], df[sentiment_col])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Calculate percentages for each category
        percentages = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

        # Calculate mean sentiment
        mean_sentiment = df.groupby("string_type")[sentiment_col].mean()

        # Perform Mann-Whitney U test
        brand_sentiments = df[df["string_type"] == "brand"][sentiment_col]
        generic_sentiments = df[df["string_type"] == "generic"][sentiment_col]
        statistic, p_value_mw = mannwhitneyu(
            brand_sentiments, generic_sentiments, alternative="two-sided"
        )

        result = {
            "Temperature": temp,
            "Chi-square Statistic": chi2,
            "Chi-square p-value": p_value,
            "Mann-Whitney U Statistic": statistic,
            "Mann-Whitney U p-value": p_value_mw,
            "Degrees of Freedom": dof,
        }

        for category in ["brand", "generic"]:
            for sentiment in [0, 1, 2]:
                label = ["Negative", "Neutral", "Positive"][sentiment]
                if category in percentages.index and sentiment in percentages.columns:
                    result[f"{category.capitalize()} {label} %"] = percentages.loc[
                        category, sentiment
                    ]
                else:
                    result[f"{category.capitalize()} {label} %"] = 0.0

            if category in mean_sentiment.index:
                result[f"{category.capitalize()} Mean Sentiment"] = mean_sentiment[
                    category
                ]
            else:
                result[f"{category.capitalize()} Mean Sentiment"] = None

        results.append(result)

    sentiment_summary = pd.DataFrame(results)
    return sentiment_summary


def process_sentiment(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
) -> pd.DataFrame:
    print(f"Processing sentiment data for {task_name}...")

    column_names = ["response_0.0", "response_0.7", "response_1.0"]
    # df = perform_sentiment_analysis(df, column_names)
    df = pd.read_parquet(
        os.path.join(output_dir, f"{task_name}_{model_name}_sentiment.parquet")
    )

    # Summarize sentiment
    sentiment_summary = summarize_sentiment(df)

    # Save sentiment summary to CSV
    summary_file = os.path.join(
        output_dir, f"{task_name}_{model_name}_sentiment_summary.csv"
    )
    sentiment_summary.to_csv(summary_file, index=False)

    # Save full sentiment data to parquet
    sentiment_file = os.path.join(
        output_dir, f"{task_name}_{model_name}_sentiment.parquet"
    )
    df.to_parquet(sentiment_file, index=False)

    print("Sentiment Summary:")
    print(sentiment_summary)

    save_to_csv(
        f"{task_name}_{model_name}_sentiment_summary.csv",
        sentiment_summary.columns.tolist(),
        sentiment_summary.values.tolist(),
    )

    return df, sentiment_summary


def plot_mean_sentiment(
    df: pd.DataFrame, output_dir: str, task_name: str, model_name: str
):
    # Reshape the data for plotting
    sentiment_summary_melted = df.melt(
        id_vars=["string_type"],
        value_vars=[
            "sentiment_response_0.0",
            "sentiment_response_0.7",
            "sentiment_response_1.0",
        ],
        var_name="temperature",
        value_name="mean_sentiment",
    )

    # Replace temperature names for better readability
    sentiment_summary_melted["temperature"] = sentiment_summary_melted[
        "temperature"
    ].str.replace("sentiment_response_", "")

    # Plotting
    temperatures = sorted(sentiment_summary_melted["temperature"].unique())
    string_types = sentiment_summary_melted["string_type"].unique()

    x = np.arange(len(temperatures))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    colors = {
        "brand": "#4a7ba7",
        "generic": "#a77b4a",
        "preferred": "#a77b4a",
    }  # Muted blue and orange

    for i, string_type in enumerate(string_types):
        subset = sentiment_summary_melted[
            sentiment_summary_melted["string_type"] == string_type
        ]
        bars = ax.bar(
            x + i * width,
            subset["mean_sentiment"],
            width,
            label=string_type.capitalize(),
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
    ax.set_ylabel("Mean Sentiment", fontweight="bold", fontsize=12)
    ax.set_title(
        f"Mean {task_name.replace('_', ' ').title()} by Temperature for {model_name}",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(temperatures, fontweight="bold")
    ax.legend(title="Drug Name Type", title_fontsize=12, fontsize=10, frameon=False)
    ax.set_ylim(0, 2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if task_name == "coral_sentiment":
        plot_dir = os.path.join(output_dir, "coral_sentiment/plots")
    else:
        plot_dir = os.path.join(output_dir, "sentiment/plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = os.path.join(plot_dir, f"{task_name}_{model_name}_sentiment_plot.png")
    plt.tight_layout()
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {plot_file}")


def calculate_sentiment_counts(data, sentiment_cols):
    sentiment_counts = {
        col: data[col].value_counts().sort_index() for col in sentiment_cols
    }

    # Ensure all sentiment categories are represented, even if they have 0 count
    for col in sentiment_cols:
        for sentiment in [0.0, 1.0, 2.0]:
            if sentiment not in sentiment_counts[col]:
                sentiment_counts[col][sentiment] = 0

    return sentiment_counts


def plot_sentiment_counts(output_dir: str, task_name: str, model_name: str):
    # Load the data
    if task_name == "coral_sentiment":
        data_file = os.path.join(
            output_dir, f"coral_sentiment/{task_name}_{model_name}_sentiment.parquet"
        )
        df = pd.read_parquet(data_file)
        df["string_type"] = df["type"]
    else:
        data_file = os.path.join(
            output_dir, f"sentiment/{task_name}_{model_name}_sentiment.parquet"
        )
        df = pd.read_parquet(data_file)

    sentiment_cols = [
        "sentiment_response_0.0",
        "sentiment_response_0.7",
        "sentiment_response_1.0",
    ]

    # Separate the data into brand names and preferred (generic) names
    brand_data = df[df["string_type"] == "brand"]
    if task_name == "coral_sentiment":
        preferred_data = df[df["string_type"] == "generic"]
    else:
        preferred_data = df[df["string_type"] == "preferred"]

    # Calculate sentiment counts for brand and preferred drugs
    sentiment_counts_brand = calculate_sentiment_counts(brand_data, sentiment_cols)
    sentiment_counts_preferred = calculate_sentiment_counts(
        preferred_data, sentiment_cols
    )

    # Data preparation for stacked bar plot
    temperatures = ["0.0", "0.7", "1.0"]

    sentiment_types = ["Negative", "Neutral", "Positive"]
    sentiment_colors = ["red", "gray", "green"]

    counts = {
        "brand": {
            temp: [
                sentiment_counts_brand[f"sentiment_response_{temp}"].get(sentiment, 0)
                for sentiment in [0.0, 1.0, 2.0]
            ]
            for temp in temperatures
        },
        "preferred": {
            temp: [
                sentiment_counts_preferred[f"sentiment_response_{temp}"].get(
                    sentiment, 0
                )
                for sentiment in [0.0, 1.0, 2.0]
            ]
            for temp in temperatures
        },
    }

    # Prepare x-axis labels combining temperature and string type
    x_labels = [f"{temp} - Brand" for temp in temperatures] + [
        f"{temp} - Preferred" for temp in temperatures
    ]
    x = np.arange(len(x_labels))  # the label locations

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35  # the width of the bars

    # print unique values
    print(f"Unique values: {df['string_type'].unique()}")
    print(f"Unique values: {df['sentiment_response_0.0'].unique()}")

    neg_counts = [counts["brand"][temp][0] for temp in temperatures] + [
        counts["preferred"][temp][0] for temp in temperatures
    ]
    neu_counts = [counts["brand"][temp][1] for temp in temperatures] + [
        counts["preferred"][temp][1] for temp in temperatures
    ]
    pos_counts = [counts["brand"][temp][2] for temp in temperatures] + [
        counts["preferred"][temp][2] for temp in temperatures
    ]

    bars_negative = ax.bar(
        x, neg_counts, width, label="Negative", color=sentiment_colors[0]
    )
    bars_neutral = ax.bar(
        x,
        neu_counts,
        width,
        bottom=neg_counts,
        label="Neutral",
        color=sentiment_colors[1],
    )
    bars_positive = ax.bar(
        x,
        pos_counts,
        width,
        bottom=[i + j for i, j in zip(neg_counts, neu_counts)],
        label="Positive",
        color=sentiment_colors[2],
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Temperature and Type")
    ax.set_ylabel("Counts")
    ax.set_title(
        f"{task_name} Distribution by Drug Type and Temperature for {model_name}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(title="Sentiment Type")

    plt.xticks(rotation=45)

    # Save the plot
    if task_name == "coral_sentiment":
        plot_dir = os.path.join(output_dir, "coral_sentiment/plots")
    else:
        plot_dir = os.path.join(output_dir, "sentiment/plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = os.path.join(
        plot_dir, f"{task_name}_{model_name}_sentiment_counts_plot.png"
    )
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to {plot_file}")

    return df


def plot_sentiment(output_dir: str, task_name: str, model_name: str):
    # Load the data
    if task_name == "coral_sentiment":
        data_file = os.path.join(
            output_dir, f"coral_sentiment/{task_name}_{model_name}_sentiment.parquet"
        )
        df = pd.read_parquet(data_file)

        # add string_type to df (type)
        df["string_type"] = df["type"]

    else:
        data_file = os.path.join(
            output_dir, f"sentiment/{task_name}_{model_name}_sentiment.parquet"
        )
        df = pd.read_parquet(data_file)

    sentiment_summary = summarize_sentiment(df)

    # Save sentiment summary to CSV
    if task_name == "coral_sentiment":
        sentiment_summary_file = os.path.join(
            output_dir, f"coral_sentiment/summary_{task_name}_{model_name}.csv"
        )
    else:
        sentiment_summary_file = os.path.join(
            output_dir, f"sentiment/summary_{task_name}_{model_name}.csv"
        )

    sentiment_summary.to_csv(sentiment_summary_file, index=False)

    print(f"Sentiment summary saved to {sentiment_summary_file}")

    # Plot the sentiment summary
    plot_mean_sentiment(sentiment_summary, output_dir, task_name, model_name)

    # Plot the sentiment counts
    plot_sentiment_counts(output_dir, task_name, model_name)

    return df
