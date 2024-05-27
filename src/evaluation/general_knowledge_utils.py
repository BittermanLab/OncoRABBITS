import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
import numpy as np


def find_correct_index(options, correct_answer):
    options = ast.literal_eval(options)
    return options.index(correct_answer) + 1  # 1-indexed


def is_correct(response, correct_index):
    try:
        response = int(response)
    except ValueError:
        return False
    return response == correct_index


def process_general_knowledge(
    df: pd.DataFrame, output_dir: str, model_name: str
) -> pd.DataFrame:
    temperatures = ["0.0", "0.7", "1.0"]
    results = []

    # Adding the correct index to the DataFrame
    df["correct_index"] = df.apply(
        lambda row: find_correct_index(row["options"], row["correct_answer"]), axis=1
    )

    # Adding correct columns for each temperature and calculating results
    for temp in temperatures:
        for string_type in ["brand name", "preferred name"]:
            correct_col = f"correct_{temp}_{string_type.replace(' ', '_')}"
            response_col = f"response_{temp}"
            df[correct_col] = df.apply(
                lambda row: (
                    is_correct(row[response_col], row["correct_index"])
                    if row["string_type"] == string_type
                    else False
                ),
                axis=1,
            )

            # Filter the DataFrame based on the string type
            filtered_df = df[df["string_type"] == string_type]

            # Calculate counts and accuracy
            correct_count = filtered_df[correct_col].sum()
            total_count = filtered_df[correct_col].count()
            accuracy = correct_count / total_count * 100 if total_count > 0 else 0

            results.append(
                {
                    "Temperature": temp,
                    "Type": string_type,
                    "Correct": correct_count,
                    "Total": total_count,
                    "Accuracy (%)": accuracy,
                }
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display the final results summary
    print("Final Results Summary:")
    print(results_df)

    # Save the final results summary to a CSV file
    results_df.to_csv(
        os.path.join(output_dir, f"{model_name}_accuracy_summary.csv"),
        index=False,
    )

    return results_df


def plot_general_knowledge(output_dir: str, model_name: str):
    # Load the data
    final_results_summary = pd.read_csv(
        os.path.join(output_dir, f"general_knowledge/{model_name}_accuracy_summary.csv")
    )
    print(final_results_summary)

    # Plot
    temperatures = final_results_summary["Temperature"].unique()
    types = final_results_summary["Type"].unique()
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    index = np.arange(len(temperatures))

    brand_data = final_results_summary[final_results_summary["Type"] == "brand name"]
    preferred_data = final_results_summary[
        final_results_summary["Type"] == "preferred name"
    ]

    ax.bar(
        index, brand_data["Accuracy (%)"], bar_width, label="Brand Name", color="blue"
    )
    ax.bar(
        index + bar_width,
        preferred_data["Accuracy (%)"],
        bar_width,
        label="Preferred Name",
        color="orange",
    )

    ax.set_title("Accuracy by Temperature and Type")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(temperatures)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    plot_dir = os.path.join(output_dir, f"general_knowledge/plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = f"{plot_dir}/accuracy_all_temps.png"
    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path)
    plt.close()

    return


# Example usage
if __name__ == "__main__":
    data = {
        "concept_code": [
            4,
            4,
            4,
            4,
            4,
            6,
            6,
            6,
            6,
            6,
            114494,
            114494,
            114494,
            114494,
            114494,
        ],
        "string_type": ["brand name"] * 5 + ["brand name"] * 5 + ["preferred name"] * 5,
        "string": ["Verzenio"] * 5 + ["Abatitor"] * 5 + ["Tabelecleucel"] * 5,
        "question": [
            "Which of the following is the brand name for the drug Abemaciclib?",
            "Which of the following is the brand name for the drug Abemaciclib?",
            "Which of the following is the brand name for the drug Abemaciclib?",
            "Which of the following is the brand name for the drug Abemaciclib?",
            "Which of the following is the brand name for the drug Abemaciclib?",
            "Which of the following is the brand name for the drug Abiraterone?",
            "Which of the following is the brand name for the drug Abiraterone?",
            "Which of the following is the brand name for the drug Abiraterone?",
            "Which of the following is the brand name for the drug Abiraterone?",
            "Which of the following is the brand name for the drug Abiraterone?",
            "Which of the following is the generic name for the drug Ebvallo?",
            "Which of the following is the generic name for the drug Ebvallo?",
            "Which of the following is the generic name for the drug Ebvallo?",
            "Which of the following is the generic name for the drug Ebvallo?",
            "Which of the following is the generic name for the drug Ebvallo?",
        ],
        "options": [
            "['Sarclisa', 'Avodart', 'Nexavar', 'Verzenio']",
            "['Abraxane', 'Emcyt', 'Faslodex', 'Verzenio']",
            "['Amnoid', 'Coxatin', 'Turalio', 'Verzenio']",
            "['Indicarb', 'Turalio', 'Yescarta', 'Verzenio']",
            "['Verzenio', 'Nidran', 'Zanosar', 'Eldesine']",
            "['Eldesine', 'Ledaga', 'Abatitor', 'Supect']",
            "['Abatitor', 'Ayvakit', 'Kymriah', 'Rydapt']",
            "['Prevacid', 'Lartruvo', 'Epidaza', 'Abatitor']",
            "['Abatitor', 'Cyendiv', 'Binorel', 'Leukine']",
            "['Didox', 'Nipent', 'Biodel', 'Abatitor']",
            "['Tabelecleucel', 'Tisagenlecleucel', 'Vincristine', 'Ponatinib']",
            "['Umbralisib', 'Cabozantinib', 'Tabelecleucel', 'Anastrozole']",
            "['Tabelecleucel', 'Talimogene laherparepvec', 'Brigatinib', 'Vemurafenib']",
            "['Medroxyprogesterone', 'Tabelecleucel', 'Paclitaxel', 'Goserelin']",
            "['Cisplatin', 'Bleomycin', 'Thioguanine', 'Tabelecleucel']",
        ],
        "correct_answer": ["Verzenio"] * 5 + ["Abatitor"] * 5 + ["Tabelecleucel"] * 5,
        "response_0.0": [4, 4, 4, 4, 1, 3, 1, 4, 2, 4, 2, 3, 3, 2, 4],
        "response_0.7": [4, 4, 4, 4, 1, 3, 1, 4, 2, 4, 2, 3, 2, 2, 4],
        "response_1.0": [4, 4, 4, 4, 1, 3, 1, 4, 2, 4, 2, 1, 4, 2, 4],
    }

    df = pd.DataFrame(data)
    output_dir = "results/"
    model_name = "example_model"
    process_general_knowledge(df, output_dir, model_name)
    plot_general_knowledge(output_dir, model_name)
