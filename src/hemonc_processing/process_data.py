import pandas as pd
import os
import random

# Directory containing the data file
data_dir = "data/drug_names"


def load_and_process_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Filter out rows that are not "preferred name" or "brand name"
    filtered_df = df[df["string_type"].isin(["preferred name", "brand name"])]

    # Group by concept_code and keep the first occurrence of each string_type per group
    unique_names_df = (
        filtered_df.groupby(["concept_code", "string_type"]).first().reset_index()
    )

    # Since we want to keep one of each type per concept_code, let's ensure there's only one of each
    final_df = unique_names_df.groupby("concept_code").filter(lambda x: len(x) <= 2)

    # Filter out concept_codes that appear only once in the DataFrame
    counts = final_df["concept_code"].value_counts()
    filtered_final_df = final_df[
        final_df["concept_code"].isin(counts[counts > 1].index)
    ]

    # Ensure we have a DataFrame that includes both a preferred name and a brand name for each concept_code
    concept_codes_with_both_names = filtered_final_df.groupby("concept_code").filter(
        lambda x: len(x) == 2
    )

    # Split the DataFrame into two: one for preferred names and one for brand names
    preferred_names_df = concept_codes_with_both_names[
        concept_codes_with_both_names["string_type"] == "preferred name"
    ]

    brand_names_df = concept_codes_with_both_names[
        concept_codes_with_both_names["string_type"] == "brand name"
    ]

    # Merge them to have a single DataFrame with both preferred and brand names for each concept_code
    combined_df = pd.merge(
        preferred_names_df,
        brand_names_df,
        on="concept_code",
        suffixes=("_preferred", "_brand"),
    )

    return combined_df


def get_all_keywords(file_path, seed=42):
    # Set the seed for the random number generator
    random.seed(seed)

    # Load the data
    df = pd.read_csv(file_path)

    # Filter out rows that are not "preferred name" or "brand name"
    filtered_df = df[df["string_type"].isin(["preferred name", "brand name"])]

    # Identify the unique generic name for each concept_code
    generic_names = filtered_df[
        filtered_df["string_type"] == "preferred name"
    ].drop_duplicates(subset="concept_code", keep="first")

    # Initialize a list to store the results
    brand_to_generic = []

    # For each concept_code, map brand names to the corresponding generic name
    for concept_code in filtered_df["concept_code"].unique():
        generic_name = generic_names[generic_names["concept_code"] == concept_code][
            "string"
        ].values[0]
        brand_names = filtered_df[
            (filtered_df["concept_code"] == concept_code)
            & (filtered_df["string_type"] == "brand name")
        ]["string"].tolist()
        for brand_name in brand_names:
            brand_to_generic.append({"brand": brand_name, "generic": generic_name})

    # Convert the results to a DataFrame
    brand_to_generic_df = pd.DataFrame(brand_to_generic)

    # generic to brand is one to many therefore we need to select a random brand name
    # here we will use the same as was selected in combined_df

    # load combined_df
    combined_df = load_and_process_data(file_path)

    # string_preferred is the generic name and string_brand is the brand name
    # select only those columns and name generic,brand -> csv
    generic_to_brand_df = combined_df[["string_preferred", "string_brand"]]
    generic_to_brand_df.columns = ["generic", "brand"]

    # Return the DataFrames
    return brand_to_generic_df, generic_to_brand_df


if __name__ == "__main__":
    # Load and process the data
    combined_df = load_and_process_data(os.path.join(data_dir, "HemOnc_drug_list.csv"))

    # Save the combined DataFrame
    combined_df.to_csv(os.path.join(data_dir, "combined_df.csv"), index=False)

    # Get all keywords
    brand_to_generic_df, generic_to_brand_df = get_all_keywords(
        os.path.join(data_dir, "HemOnc_drug_list.csv")
    )

    # save keywords
    brand_to_generic_df.to_csv(
        os.path.join(data_dir, "brand_to_generic_df.csv"), index=False
    )

    generic_to_brand_df.to_csv(
        os.path.join(data_dir, "generic_to_brand_df.csv"), index=False
    )
