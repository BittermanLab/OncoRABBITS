import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to print statistics for each type and temperature
def save_statistics(df, model, task_name, output_dir):
    statistics = []
    for type_ in ["brand", "generic"]:
        for temp in ["0.0", "0.7", "1.0"]:
            temp_col = f"response_{temp}"
            df_temp = df[(df["type"] == type_) & (~df[temp_col].isnull())]

            # Try to convert response to integer, set to None if conversion fails
            df_temp[temp_col] = pd.to_numeric(df_temp[temp_col], errors="coerce")

            # Calculate statistics
            valid_responses = df_temp[temp_col].dropna()
            stats = {
                "model": model,
                "task": task_name,
                "type": type_,
                "temperature": temp,
                "mean_length": valid_responses.mean(),
                "median_length": valid_responses.median(),
                "std_dev": valid_responses.std(),
                "min_length": valid_responses.min(),
                "max_length": valid_responses.max(),
                "total_responses": len(df_temp),
                "valid_responses": len(valid_responses),
                "na_responses": df_temp[temp_col].isna().sum(),
            }
            statistics.append(stats)

    stats_df = pd.DataFrame(statistics)
    stats_dir = os.path.join(output_dir, f"{model}/{task_name}")
    os.makedirs(stats_dir, exist_ok=True)

    stats_file_path = os.path.join(stats_dir, "statistics.csv")

    stats_df.to_csv(stats_file_path, index=False)
    print(f"Saved statistics for {model} {task_name} to {stats_file_path}")

    # Plotting
    plot_statistics(stats_df, model, task_name, output_dir)
    print(f"Plots saved")


def plot_statistics(stats_df, model, task_name, output_dir):
    temperatures = ["0.0", "0.7", "1.0"]
    types = ["brand", "generic"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_title(f"Mean Length with Std Dev for {task_name}")
    width = 0.3  # the width of the bars

    for j, type_ in enumerate(types):
        type_df = stats_df[stats_df["type"] == type_]
        positions = np.arange(len(temperatures)) + (width * j)
        ax.bar(
            positions,
            type_df["mean_length"],
            yerr=type_df["std_dev"],
            width=width,
            label=type_,
            alpha=0.7,
            capsize=5,
        )

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean Length")
    ax.set_xticks(np.arange(len(temperatures)) + width / 2)
    ax.set_xticklabels(temperatures)
    ax.legend()

    plot_dir = os.path.join(output_dir, f"{model}/{task_name}")
    os.makedirs(plot_dir, exist_ok=True)

    plot_file_path = os.path.join(plot_dir, "stats.png")
    plt.savefig(plot_file_path)
    plt.close()
    print(f"Saved plot for {model} {task_name} to {plot_file_path}")
