import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("MarieAngeA13/Sentiment-Analysis-BERT")
model = AutoModelForSequenceClassification.from_pretrained(
    "MarieAngeA13/Sentiment-Analysis-BERT"
)


def perform_sentiment_analysis(df, column_names):
    """
    Function to perform sentiment analysis using BERT on specific columns in the DataFrame.
    """
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
    Function to summarize sentiment values for brand names and preferred names.
    """
    sentiment_summary = (
        df.groupby("string_type")[
            [
                "sentiment_response_0.0",
                "sentiment_response_0.7",
                "sentiment_response_1.0",
            ]
        ]
        .mean()
        .reset_index()
    )
    return sentiment_summary


def process_sentiment(
    df: pd.DataFrame, output_dir: str, task_name: str
) -> pd.DataFrame:
    print(f"Processing sentiment data for {task_name}...")

    column_names = ["response_0.0", "response_0.7", "response_1.0"]
    df = perform_sentiment_analysis(df, column_names)

    # save sentiment data to parquet
    sentiment_file = os.path.join(
        output_dir, f"sentiment/{task_name}_sentiment.parquet"
    )

    df.to_parquet(sentiment_file, index=False)

    sentiment_summary = summarize_sentiment(df)

    # Save sentiment summary to CSV
    sentiment_summary_file = os.path.join(
        output_dir, f"sentiment/summary_{task_name}.csv"
    )
    sentiment_summary.to_csv(sentiment_summary_file, index=False)

    print(f"Sentiment summary saved to {sentiment_summary_file}")

    return df
