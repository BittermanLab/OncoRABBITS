# symptoms derived from https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.01440

import logging
import re
import json
from collections import Counter
import pandas as pd


patient_questions = [
    "Most recent clinic note: {clinic_note}\n\nPatient's reported symptoms: {symptoms}\n\nBased on the patient's current use of {drug}, how likely is it that these symptoms represent an immune-related adverse event (irAE)? Please return the corresponding number as: 1 (very unlikely), 2 (unlikely), 3 (likely), 4 (very likely).",
    "Most recent clinic note: {clinic_note}\n\nPatient's reported symptoms: {symptoms}\n\nConsidering the patient's history and current medications, what are the top 3 most likely diagnoses causing these symptoms? Please list them in order of probability.",
]

irAE_symptoms = {
    "Cutaneous toxicities": [
        "I've developed an itchy rash all over my body",
        "I've noticed some new skin bumps that aren't itchy",
        "The raised patches on my skin seem to be getting worse",
        "I'm seeing more and more flat, discolored areas on my skin",
        "The scaly patches on my skin are spreading",
        "I've noticed some areas where my skin color is fading",
        "I have some fluid-filled blisters on my skin",
        "I keep getting hives that won't go away",
        "I have a fever and a widespread rash, and my skin feels painful and is peeling",
        "My face looks swollen, especially around my eyes",
    ],
    "GI toxicities": [
        "I've been having stomach pain, feeling nauseous, and having frequent diarrhea",
        "I have a fever and I've noticed blood and mucus in my stool",
        "I've been throwing up, I don't feel like eating, and my skin and eyes look yellow",
        "I have pain in my abdomen and I've noticed my urine is darker than usual",
        "My gums bleed when I brush my teeth, which isn't normal for me",
        "I've been bruising really easily lately, even from minor bumps",
    ],
    "Lung toxicities": [
        "I've developed a cough that won't go away",
        "I'm feeling short of breath, even when I'm not doing much",
        "I have pain in my chest and a fever",
    ],
    "Endocrine toxicities": [
        "I've been having headaches and my vision seems off",
        "I've been feeling unusually cold all the time",
        "I'm exhausted all the time and I've noticed changes in my weight",
        "I've been feeling dizzy and lightheaded",
        "I'm urinating much more frequently and I'm always thirsty",
    ],
    "Musculoskeletal toxicities": [
        "My joints are painful and swollen",
        "My muscles are sore and I feel weaker than usual",
        "The muscles in my shoulders and hips are stiff and painful",
    ],
    "Renal toxicities": [
        "I've noticed changes in my urination and I feel bloated",
        "I have pain in my lower abdomen",
        "I feel nauseous, I'm vomiting, and my blood pressure is high",
        "I feel confused and not quite like myself",
    ],
    "Nervous system toxicities": [
        "I'm feeling weak and I'm experiencing unusual sensations in my body",
        "I have a bad headache and I'm not thinking clearly",
    ],
    "Hematologic toxicities": [
        "I've been feeling unusually tired and short of breath",
        "I've noticed some unusual bruising on my body",
        "I've noticed some unusual bleeding recently",
        "I've been having fevers and night sweats",
    ],
    "Cardiovascular toxicities": [
        "I'm always tired and my muscles ache",
        "My heart feels like it's racing",
        "I have some new chest pain",
        "I feel lightheaded and short of breath",
        "My legs and feet are swollen",
        "I can't walk as far as I used before running out of breath",
    ],
    "Ocular toxicities": [
        "My vision is blurry and seems to be changing",
        "My eyes are really sensitive to light",
        "My eyes are painful, red, and dry",
    ],
}


def extract_text_before_patient_message(df):
    df["Extracted_Text"] = df["Input"].apply(
        lambda x: x.split("Patient message:")[0].strip()
    )
    return df[["pin", "Extracted_Text"]]


def custom_parse_response(response):
    """
    Custom function to parse drug list from response string, with improved handling
    for control characters.
    """
    if isinstance(response, str):
        try:
            # Remove leading and trailing square brackets
            response = response.strip("[]")
            # Split the response by commas not within quotes
            parts = re.split(r'(?<!"),(?!")', response)
            # Strip whitespace and add double quotes around parts if not already quoted
            parts = [part.strip().strip('"') for part in parts]
            # Properly escape control characters in each part
            parts = [json.dumps(part) for part in parts]
            # Join parts back with commas
            response = ",".join(parts)
            # Form the final JSON array
            response = f"[{response}]"
            # Parse the JSON
            drug_list = json.loads(response)
            return drug_list
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing response: {response}, error: {e}")
            return []
    else:
        return []


def extract_keywords_from_col(df, col, keywords):
    def extract_keywords(col_value, keywords):
        found_keywords = []
        if col_value is None or (
            isinstance(col_value, (str, float)) and pd.isna(col_value)
        ):
            return found_keywords

        keywords = sorted(keywords, key=len, reverse=True)

        if isinstance(col_value, list):
            for item in col_value:
                for keyword in keywords:
                    if re.search(
                        rf"\b{re.escape(keyword)}\b", str(item), re.IGNORECASE
                    ):
                        found_keywords.append(keyword)
        elif isinstance(col_value, str):
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", col_value, re.IGNORECASE):
                    found_keywords.append(keyword)

        logging.debug(f"Extracted keywords from '{col_value}': {found_keywords}")
        return list(set(found_keywords))

    new_col_name = f"{col}_keywords"
    df[new_col_name] = df[col].apply(
        lambda x: extract_keywords(custom_parse_response(x), keywords)
    )

    keyword_counts = Counter()
    for keywords_list in df[new_col_name]:
        keyword_counts.update(keywords_list)

    return keyword_counts, df


def filter_df_by_keywords(df, keyword_col, filter_keywords):
    filter_pattern = re.compile(
        r"|".join(map(re.escape, filter_keywords)), re.IGNORECASE
    )
    filtered_df = df[
        df[keyword_col].apply(
            lambda keywords: any(filter_pattern.search(keyword) for keyword in keywords)
        )
    ]
    return filtered_df


def aggregate_keyword_counts(counts1):
    # Convert Counter objects to DataFrames
    df1 = (
        pd.DataFrame.from_dict(counts1, orient="index", columns=["response_count"])
        .reset_index()
        .rename(columns={"index": "Keyword"})
    )

    # Convert counts back to integers (they were converted to floats by fillna)
    df1["response_count"] = df1["response_count"].fillna(0).astype(int)
    return df1


def generate_prompts(df, irAE_symptoms):
    prompts = []

    for _, row in df.iterrows():
        context = row["Extracted_Text"]
        drugs = row["Extracted_Text_keywords"]

        # Ensure drugs is a list
        if isinstance(drugs, str):
            drugs = [
                drug.strip() for drug in drugs.strip("[]").split(",") if drug.strip()
            ]
        elif not isinstance(drugs, list):
            drugs = []

        for drug in drugs:
            for symptom_category, symptom_list in irAE_symptoms.items():
                for symptom in symptom_list:
                    for question in patient_questions:
                        prompt = question.format(
                            clinic_note=context, symptoms=symptom, drug=drug
                        )
                        prompts.append(
                            {
                                "pin": row["pin"],
                                "drug": drug,
                                "symptom_category": symptom_category,
                                "symptom": symptom,
                                "prompt": prompt,
                            }
                        )

    return pd.DataFrame(prompts)


if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_colwidth", None)

    ######### Process OCQA data #########
    print("Processing OncQA data...")

    # Read in src/irAE/ocqa.csv
    oncqa = pd.read_csv("src/irAE/oncqa.csv")

    # Extract text before "patient message"
    oncqa = extract_text_before_patient_message(oncqa)

    ###### Get drug counts ######

    # Load all keywords
    from drug_mapping import DrugMapper

    mapper = DrugMapper(
        "data/drug_names/brand_to_generic_df.csv",
        "data/drug_names/generic_to_brand_df.csv",
    )
    all_keywords = mapper.load_all_keywords_list()

    # Extract keyword counts from the Extracted_Text column
    full_response_counts, df_keywords = extract_keywords_from_col(
        oncqa, "Extracted_Text", all_keywords
    )

    # Save full DataFrame with keywords to CSV
    df_keywords.to_csv("src/irAE/full_df_keywords.csv", index=False)
    print(df_keywords.head())

    # Calculate full drug counts
    full_drug_counts = aggregate_keyword_counts(full_response_counts)
    full_drug_counts = full_drug_counts.sort_values(
        by="response_count", ascending=False
    )

    # Save full drug counts to CSV
    full_drug_counts.to_csv("src/irAE/full_drug_counts.csv", index=False)

    ######### Read in immunotherapy data #########
    print("Processing immunotherapy data...")
    immunotherapy = pd.read_csv("src/irAE/immunotherapy_list.csv")

    print("Original data:")
    print(immunotherapy.head())
    print(f"Original shape: {immunotherapy.shape}")

    # Remove first row after header (if it's still necessary)
    immunotherapy = immunotherapy.iloc[1:]

    # Convert columns 4-6 to boolean, ensuring "TRUE" is True and everything else is False
    boolean_columns = [
        "Valid Immunotherapy",
        "First line",
        "No prior systemic exposure",
    ]
    for col in boolean_columns:
        immunotherapy[col] = immunotherapy[col].map({"TRUE": True}).fillna(False)

    # Filter rows where all 3 conditions are True
    immunotherapy_filtered = immunotherapy[immunotherapy[boolean_columns].all(axis=1)]

    # add pin column as unique identifier
    immunotherapy_filtered["pin"] = range(1, len(immunotherapy_filtered) + 1)

    print("\nFiltered data:")
    print(immunotherapy_filtered.head())
    print(f"Filtered shape: {immunotherapy_filtered.shape}")

    # Save to csv
    immunotherapy_filtered.to_csv(
        "src/irAE/immunotherapy_list_filtered.csv", index=False
    )
    print("\nFiltered data saved to 'src/irAE/immunotherapy_list_filtered.csv'")

    ####### Filter OncQA data ######
    print("Filtering OncQA data...")

    # Filter rows where keywords are in immunotherapy_list_filtered
    filtered_oncqa = filter_df_by_keywords(
        df_keywords, "Extracted_Text_keywords", immunotherapy["Immunotherapy"]
    )

    # Save filtered DataFrame to CSV
    filtered_oncqa.to_csv("src/irAE/filtered_oncqa.csv", index=False)

    # Calculate drug counts for the filtered DataFrame
    filtered_response_counts = Counter()
    for keywords_list in filtered_oncqa["Extracted_Text_keywords"]:
        filtered_response_counts.update(keywords_list)

    # Aggregate the filtered counts
    filtered_drug_counts = aggregate_keyword_counts(filtered_response_counts)
    filtered_drug_counts = filtered_drug_counts.sort_values(
        by="response_count", ascending=False
    )

    # Save the filtered count to a CSV file
    filtered_drug_counts.to_csv("src/irAE/filtered_drug_counts.csv", index=False)

    # Save the filtered DataFrame with keywords to CSV
    filtered_oncqa.to_csv("src/irAE/filtered_df_keywords.csv", index=False)

    ### Create IrAEQA data ####

    print("Oncqa dataset shape:", filtered_oncqa.shape)
    print("Immunotherapy dataset shape:", immunotherapy_filtered.shape)

    # Prepare the immunotherapy dataset
    immunotherapy_subset = immunotherapy_filtered[
        ["pin", "Example EHR Context", "Immunotherapy", "Regime"]
    ].copy()
    immunotherapy_subset.rename(
        columns={
            "Example EHR Context": "Extracted_Text",
            "Immunotherapy": "Extracted_Text_keywords",
        },
        inplace=True,
    )

    # Ensure filtered_oncqa has the same columns as immunotherapy_subset
    filtered_oncqa["Regime"] = ""  # Add empty 'Regime' column to filtered_oncqa

    # Reorder columns in filtered_oncqa to match immunotherapy_subset
    filtered_oncqa = filtered_oncqa[
        ["pin", "Extracted_Text", "Extracted_Text_keywords", "Regime"]
    ]

    # Stack the datasets
    stacked_data = pd.concat(
        [filtered_oncqa, immunotherapy_subset], axis=0, ignore_index=True
    )

    print("\nStacked dataset shape:", stacked_data.shape)
    print("\nStacked dataset columns:")
    print(stacked_data.columns)

    # Display a few rows of the stacked dataset
    print("\nSample of stacked dataset:")
    print(stacked_data.head())

    # Save to csv
    stacked_data.to_csv("src/irAE/merged_df.csv", index=False)
    print("\nStacked dataset saved to 'src/irAE/merged_df.csv'")

    ###### Generate prompts ######
    # filtered_oncqa = pd.read_csv("src/irAE/filtered_oncqa.csv")
    prompts_df = generate_prompts(stacked_data, irAE_symptoms)

    # Save the prompts to a CSV file
    prompts_df.to_csv("src/irAE/generated_prompts.csv", index=False)

    # Print a few examples
    print(prompts_df.head())
