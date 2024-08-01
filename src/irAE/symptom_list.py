# symptoms derived from https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.01440

import logging
import re
import json
from collections import Counter
import pandas as pd
from drug_mapping import DrugMapper


patient_questions = {
    "irae_detection": "Most recent clinic note: {clinic_note}\n\nPatient's reported symptoms: {symptoms}\n\nBased on the patient's current use of {drug}, how likely is it that these symptoms represent an immune-related adverse event (irAE)? Please return the corresponding number as: 1 (very unlikely), 2 (unlikely), 3 (likely), 4 (very likely). Answer: ",
    "differential": "Most recent clinic note: {clinic_note}\n\nPatient's reported symptoms: {symptoms}\n\nConsidering the patient's history and current medications, what are the top 3 most likely diagnoses causing these symptoms? Please list them in order of probability. Answer: [",
}

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


def create_brand_only_version(df: pd.DataFrame, mapper: DrugMapper) -> pd.DataFrame:
    """
    Create a brand-only version of the dataset by replacing generic names with brand names.
    """
    generic_to_brand = mapper.load_keywords("generic_to_brand")

    def replace_generic_with_brand(text):
        for generic, brand in generic_to_brand.items():
            text = re.sub(
                r"\b" + re.escape(generic) + r"\b", brand, text, flags=re.IGNORECASE
            )
        return text

    brand_only_df = df.copy()
    brand_only_df["Extracted_Text"] = brand_only_df["Extracted_Text"].apply(
        replace_generic_with_brand
    )
    brand_only_df["Extracted_Text_keywords"] = brand_only_df[
        "Extracted_Text_keywords"
    ].apply(lambda x: [replace_generic_with_brand(drug) for drug in x])

    return brand_only_df


def create_generic_only_version(df: pd.DataFrame, mapper: DrugMapper) -> pd.DataFrame:
    """
    Create a generic-only version of the dataset by replacing brand names with generic names.
    """
    brand_to_generic = mapper.load_keywords("brand_to_generic")

    def replace_brand_with_generic(text):
        for brand, generic in brand_to_generic.items():
            text = re.sub(
                r"\b" + re.escape(brand) + r"\b", generic, text, flags=re.IGNORECASE
            )
        return text

    generic_only_df = df.copy()
    generic_only_df["Extracted_Text"] = generic_only_df["Extracted_Text"].apply(
        replace_brand_with_generic
    )
    generic_only_df["Extracted_Text_keywords"] = generic_only_df[
        "Extracted_Text_keywords"
    ].apply(lambda x: [replace_brand_with_generic(drug) for drug in x])

    return generic_only_df


def extract_text_before_patient_message(df):
    df["Extracted_Text"] = df["Input"].apply(
        lambda x: x.split("Patient message:")[0].strip()
    )
    return df[["pin", "Extracted_Text"]]


def custom_parse_response(response):
    """
    Custom function to parse drug list from response string, ensuring whole words are extracted.
    """
    if isinstance(response, str):
        try:
            # Remove leading and trailing square brackets
            response = response.strip("[]")
            # Split the response by commas not within quotes
            parts = re.split(r'(?<!"),(?!")', response)
            # Strip whitespace and quotes from each part
            parts = [part.strip().strip('"').strip("'") for part in parts]
            # Remove any empty strings and join characters into whole words
            parts = ["".join(part.split()) for part in parts if part]
            return parts
        except Exception as e:
            print(f"Error parsing response: {response}, error: {e}")
            return []
    elif isinstance(response, list):
        return ["".join(item.split()) for item in response if item]
    else:
        return []


def extract_keywords_from_col(df, col, keywords, mapper):
    def extract_keywords(col_value, keywords):
        found_keywords = []
        if col_value is None or (
            isinstance(col_value, (str, float)) and pd.isna(col_value)
        ):
            return found_keywords

        keywords = sorted(keywords, key=len, reverse=True)

        if isinstance(col_value, list):
            col_value = " ".join(col_value)  # Join list elements into a single string

        if isinstance(col_value, str):
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", col_value, re.IGNORECASE):
                    found_keywords.append(keyword)

        logging.debug(f"Extracted keywords from '{col_value}': {found_keywords}")
        return list(set(found_keywords))

    new_col_name = f"{col}_keywords"
    df[new_col_name] = df[col].apply(
        lambda x: extract_keywords(custom_parse_response(x), keywords)
    )

    # Add new columns for brand and generic drug names
    brand_to_generic = mapper.load_keywords("brand_to_generic")
    generic_to_brand = mapper.load_keywords("generic_to_brand")

    def get_alternative_names(drug_list):
        brand_names = []
        generic_names = []
        for drug in drug_list:
            drug_lower = drug.lower()
            if drug_lower in brand_to_generic:
                generic_names.append(brand_to_generic[drug_lower])
                brand_names.append(drug)
            elif drug_lower in generic_to_brand:
                brand_names.append(generic_to_brand[drug_lower])
                generic_names.append(drug)
            else:
                brand_names.append(drug)
                generic_names.append(drug)
        return brand_names, generic_names

    df["brand_names"], df["generic_names"] = zip(
        *df[new_col_name].apply(get_alternative_names)
    )

    keyword_counts = Counter()
    for keywords_list in df[new_col_name]:
        keyword_counts.update(keywords_list)

    return keyword_counts, df


def map_immunotherapy_drugs(df, mapper):
    brand_to_generic = mapper.load_keywords("brand_to_generic")
    generic_to_brand = mapper.load_keywords("generic_to_brand")

    def map_drugs(drug_list):
        brand_names = []
        generic_names = []
        for drug in drug_list:
            drug_lower = drug.lower()
            if drug_lower in brand_to_generic:
                brand_names.append(drug)
                generic_names.append(brand_to_generic[drug_lower])
            elif drug_lower in generic_to_brand:
                generic_names.append(drug)
                brand_names.append(generic_to_brand[drug_lower])
            else:
                brand_names.append(drug)
                generic_names.append(drug)
        return list(set(brand_names)), list(set(generic_names))

    df["brand_names"], df["generic_names"] = zip(*df["Immunotherapy"].apply(map_drugs))
    return df


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
                for symptom_index, symptom in enumerate(symptom_list):
                    for question_key, question_template in patient_questions.items():
                        prompt = question_template.format(
                            clinic_note=context, symptoms=symptom, drug=drug
                        )
                        # Generate unique ID
                        unique_id = f"{row['pin']}_{drug}_{symptom_category}_{symptom_index}_{question_key}"
                        prompts.append(
                            {
                                "unique_id": unique_id,
                                "pin": row["pin"],
                                "type": row["type"],
                                "task_name": question_key,
                                "drug": drug,
                                "symptom_category": symptom_category,
                                "symptom": symptom,
                                "prompt": prompt,
                            }
                        )
    return pd.DataFrame(prompts)


def apply_drug_mapping(df, mapper, text_column, keywords_column):
    """
    Apply drug mapping to a DataFrame, adding brand and generic name columns.
    """
    all_keywords = mapper.load_all_keywords_list()
    _, df_with_keywords = extract_keywords_from_col(
        df, text_column, all_keywords, mapper
    )

    return df_with_keywords


if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_colwidth", None)

    # Initialize DrugMapper
    mapper = DrugMapper(
        "data/drug_names/brand_to_generic_df.csv",
        "data/drug_names/generic_to_brand_df.csv",
    )

    ######### Process OCQA data #########
    print("Processing OncQA data...")
    oncqa = pd.read_csv("src/irAE/oncqa.csv")
    oncqa = extract_text_before_patient_message(oncqa)
    oncqa_with_drugs = apply_drug_mapping(
        oncqa, mapper, "Extracted_Text", "Extracted_Text_keywords"
    )

    all_keywords = mapper.load_all_keywords_list()

    # Calculate full drug counts
    full_response_counts, df_keywords = extract_keywords_from_col(
        oncqa_with_drugs, "Extracted_Text", all_keywords, mapper
    )

    full_drug_counts = aggregate_keyword_counts(full_response_counts)
    full_drug_counts = full_drug_counts.sort_values(
        by="response_count", ascending=False
    )

    ######### Read in immunotherapy data #########
    print("Processing immunotherapy data...")
    immunotherapy = pd.read_csv("src/irAE/immunotherapy_list.csv")

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

    # Split the Immunotherapy column into a list of drugs
    immunotherapy_filtered["Immunotherapy"] = immunotherapy_filtered[
        "Immunotherapy"
    ].apply(lambda x: [drug.strip() for drug in x.split("+")])

    # Apply our new mapping function
    immunotherapy_with_drugs = map_immunotherapy_drugs(immunotherapy_filtered, mapper)

    # Print a sample of the results
    print(
        immunotherapy_with_drugs[
            ["Immunotherapy", "brand_names", "generic_names"]
        ].head()
    )

    # Save the results
    immunotherapy_with_drugs.to_csv(
        "src/irAE/immunotherapy_with_mapped_drugs.csv", index=False
    )

    ####### Filter OncQA data ######
    print("Filtering OncQA data...")
    filtered_oncqa = filter_df_by_keywords(
        oncqa_with_drugs, "Extracted_Text_keywords", immunotherapy["Immunotherapy"]
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

    print("Filtered oncqa:")
    print(filtered_oncqa)
    print("\n")
    print("Immunotherapy filtered:")
    print(immunotherapy_filtered)
    print("Filtered")

    #### THIS ALL WORKS ABOVE - use filtered_oncqa to join to ####

    ### Create IrAEQA data ####
    print("Oncqa dataset shape:", filtered_oncqa.shape)
    print("Immunotherapy dataset shape:", immunotherapy_with_drugs.shape)

    # add type col to immunotherapy_filtered
    immunotherapy_with_drugs["type"] = "immunotherapy"
    filtered_oncqa["type"] = "oncqa"

    # Prepare the immunotherapy dataset
    immunotherapy_subset = immunotherapy_with_drugs[
        [
            "pin",
            "Example EHR Context",
            "Immunotherapy",
            "Regime",
            "type",
            "brand_names",
            "generic_names",
        ]
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
        [
            "pin",
            "Extracted_Text",
            "Extracted_Text_keywords",
            "Regime",
            "type",
            "brand_names",
            "generic_names",
        ]
    ]

    # Stack the datasets
    stacked_data = pd.concat(
        [filtered_oncqa, immunotherapy_subset], axis=0, ignore_index=True
    )

    # Display a few rows of the stacked dataset from row 4-7
    print("\nSample of stacked dataset:")
    print(stacked_data.iloc[2:8])

    # Save to csv
    stacked_data.to_csv("src/irAE/merged_df.csv", index=False)
    print("\nStacked dataset saved to 'src/irAE/merged_df.csv'")

    # Create brand-only version
    brand_only_data = create_brand_only_version(stacked_data, mapper)
    brand_only_data.to_csv("src/irAE/merged_df_brand_only.csv", index=False)
    print("\nBrand-only dataset saved to 'src/irAE/merged_df_brand_only.csv'")

    # Create generic-only version
    generic_only_data = create_generic_only_version(stacked_data, mapper)
    generic_only_data.to_csv("src/irAE/merged_df_generic_only.csv", index=False)
    print("\nGeneric-only dataset saved to 'src/irAE/merged_df_generic_only.csv'")

    ###### Generate prompts ######
    # Generate prompts for original stacked data
    prompts_df = generate_prompts(stacked_data, irAE_symptoms)
    prompts_df.to_csv("src/irAE/generated_prompts.csv", index=False)
    print("\nPrompts for original data saved to 'src/irAE/generated_prompts.csv'")

    # Generate prompts for brand-only version
    brand_only_prompts_df = generate_prompts(brand_only_data, irAE_symptoms)
    brand_only_prompts_df.to_csv(
        "src/irAE/generated_prompts_brand_only.csv", index=False
    )
    print(
        "\nPrompts for brand-only data saved to 'src/irAE/generated_prompts_brand_only.csv'"
    )

    # Generate prompts for generic-only version
    generic_only_prompts_df = generate_prompts(generic_only_data, irAE_symptoms)
    generic_only_prompts_df.to_csv(
        "src/irAE/generated_prompts_generic_only.csv", index=False
    )
    print(
        "\nPrompts for generic-only data saved to 'src/irAE/generated_prompts_generic_only.csv'"
    )

    # Print a few examples from each dataset
    print("\nSample of original prompts:")
    print(prompts_df.head())
    print("\nSample of brand-only prompts:")
    print(brand_only_prompts_df.head())
    print("\nSample of generic-only prompts:")
    print(generic_only_prompts_df.head())
