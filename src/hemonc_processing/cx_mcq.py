import pandas as pd
import numpy as np
import random
from typing import List, Dict, Union, Tuple
from collections import defaultdict


def filter_drug_data(data: pd.DataFrame, drugs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the data to ensure that if x_type or y_type is 'drug',
    the corresponding name is in the brand or generic list.
    Also add columns to track whether the drug name is brand or generic.
    """
    # Create dictionaries for faster lookup
    brand_to_generic = dict(zip(drugs_df["brand"], drugs_df["generic"]))
    generic_to_brand = dict(zip(drugs_df["generic"], drugs_df["brand"]))
    all_drugs = set(brand_to_generic.keys()).union(set(generic_to_brand.keys()))

    def check_drug(name):
        if name in brand_to_generic:
            return True, "brand"
        elif name in generic_to_brand:
            return True, "generic"
        return False, None

    # Vectorized function for checking drug presence and type
    vec_check_drug = np.vectorize(check_drug)

    # Apply the function to create new columns
    data["x_in_list"], data["x_drug_type"] = vec_check_drug(data["x_name"])
    data["y_in_list"], data["y_drug_type"] = vec_check_drug(data["y_name"])

    # Create boolean masks
    x_is_drug = data["x_type"] == "drug"
    y_is_drug = data["y_type"] == "drug"

    # Combine masks
    mask = (
        (x_is_drug & data["x_in_list"] & ~y_is_drug)
        | (y_is_drug & data["y_in_list"] & ~x_is_drug)
        | (x_is_drug & y_is_drug & data["x_in_list"] & data["y_in_list"])
    )

    return data[mask]


def display_relation(
    data: pd.DataFrame, target_relation: str, brand_to_generic: Dict[str, str]
) -> List[Dict[str, Union[str, List[str]]]]:
    # Filter data for the target relation
    filtered_data = data[data["display_relation"] == target_relation]

    # Get unique combinations of x_type and y_type
    type_combinations = filtered_data[["x_type", "y_type"]].drop_duplicates().values

    generic_mcqs = []

    for x_type, y_type in type_combinations:
        # Filter data for this specific type combination
        type_filtered_data = filtered_data[
            (filtered_data["x_type"] == x_type) & (filtered_data["y_type"] == y_type)
        ]

        # Group data by relation type using a defaultdict
        relations = defaultdict(list)
        for _, row in type_filtered_data.iterrows():
            relations[row["relation"]].append(
                (row["x_name"], row["y_name"], row["x_drug_type"], row["y_drug_type"])
            )

        # Generate MCQs for this type combination
        for relation, items in relations.items():
            # Create sets of all y_names for faster lookup
            all_generic_y_names = set()
            for _, y_name, _, y_drug_type in items:
                if y_drug_type == "generic":
                    all_generic_y_names.add(y_name)
                else:
                    all_generic_y_names.add(brand_to_generic.get(y_name, y_name))

            for x_name, y_name, x_drug_type, y_drug_type in items:
                # Convert to generic names
                generic_x = (
                    brand_to_generic.get(x_name, x_name)
                    if x_drug_type == "brand"
                    else x_name
                )
                generic_y = (
                    brand_to_generic.get(y_name, y_name)
                    if y_drug_type == "brand"
                    else y_name
                )

                generic_mcq = generate_mcq(
                    generic_x, generic_y, all_generic_y_names, relation, x_type, y_type
                )

                if generic_mcq:
                    generic_mcqs.append(generic_mcq)

    return generic_mcqs


def generate_mcq(x_name, y_name, all_y_names, relation, x_type, y_type):
    other_y_names = list(all_y_names - {y_name})

    if len(other_y_names) < 3:
        return None

    incorrect_options = random.sample(other_y_names, 3)
    options = [y_name] + incorrect_options
    random.shuffle(options)

    return {
        "relation": relation,
        "question_option": x_name,
        "options": options,
        "correct_answer": y_name,
        "x_type": x_type,
        "y_type": y_type,
        "type_direction": f"{x_type} -> {y_type}",
    }


def process_target_relation(
    target_relation: str, primekg: pd.DataFrame, drugs: pd.DataFrame
):
    print(f"Processing {target_relation}...")
    initial_data = primekg[primekg["display_relation"] == target_relation]
    filtered_data = filter_drug_data(initial_data, drugs)
    # save txt with "Number of relations: "
    with open(f"data/questions/{target_relation}_filtered_data.txt", "w") as f:
        f.write(f"Number of relations: {len(filtered_data)}\n")

    filtered_data = filtered_data.sample(1000)

    brand_to_generic = dict(zip(drugs["brand"], drugs["generic"]))

    generic_mcq_list = display_relation(
        filtered_data, target_relation, brand_to_generic
    )
    generic_mcq_df = pd.DataFrame(generic_mcq_list)
    return generic_mcq_df


# Function to create the question template
def create_question(row):
    if "type_direction" in row and row["type_direction"] == "drug -> disease":
        return f"Which of the following diseases are contraindicated with the drug {row['question_option']}?"
    else:
        return f"Which of the following drugs are contraindicated with {row['question_option']}?"


# Function to find the correct label index
def find_correct_label(row):
    for i in range(4):
        if row[f"ending{i}"] == row["correct_answer"]:
            return i
    return -1  # Return -1 if no match is found (shouldn't happen if data is consistent)


def create_brand_version(generic_df, generic_to_brand):
    brand_df = generic_df.copy()

    def replace_with_brand(name):
        return generic_to_brand.get(name, name)

    for column in ["question_option", "correct_answer"] + [
        f"ending{i}" for i in range(4)
    ]:
        if column in brand_df.columns:
            brand_df[column] = brand_df[column].apply(replace_with_brand)

    if "sent1" in brand_df.columns:
        brand_df["sent1"] = brand_df.apply(create_question, axis=1)

    return brand_df


if __name__ == "__main__":
    # Read in brand_to_generic_df.csv
    drugs = pd.read_csv("data/drug_names/brand_to_generic_df.csv")

    # run this in the terminal if needed
    # wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620
    primekg = pd.read_csv("src/hemonc_processing/kg.csv", low_memory=False)

    # Assuming primekg is already loaded
    generic_contraindication_df = process_target_relation(
        "synergistic interaction", primekg, drugs
    )

    # Prepare the generic dataset
    generic_contraindication_df["sent1"] = generic_contraindication_df.apply(
        create_question, axis=1
    )
    for i in range(4):
        generic_contraindication_df[f"ending{i}"] = generic_contraindication_df[
            "options"
        ].apply(lambda x: x[i])
    generic_contraindication_df["label"] = generic_contraindication_df.apply(
        find_correct_label, axis=1
    )

    generic_questions = generic_contraindication_df[
        [
            "sent1",
            "question_option",
            "ending0",
            "ending1",
            "ending2",
            "ending3",
            "label",
            "correct_answer",
            "type_direction",
        ]
    ]

    # Create brand version
    generic_to_brand = dict(zip(drugs["generic"], drugs["brand"]))
    brand_questions = create_brand_version(generic_questions, generic_to_brand)

    # Save the new datasets to parquet files
    generic_questions.to_parquet(
        "data/questions/drugs_contraindications_generic.parquet", index=False
    )
    brand_questions.to_parquet(
        "data/questions/drugs_contraindications_brand.parquet", index=False
    )

    print("Parquet files created successfully.")

    # Verify that the questions are indexed the same way
    print("\nVerification:")
    print(f"Number of generic questions: {len(generic_questions)}")
    print(f"Number of brand questions: {len(brand_questions)}")
    print("\nSample comparison:")
    for i in range(min(5, len(generic_questions))):
        print(f"\nQuestion {i}:")
        print(f"Generic: {generic_questions.iloc[i]['sent1']}")
        print(f"Brand: {brand_questions.iloc[i]['sent1']}")
        print(
            "Generic options:",
            generic_questions.iloc[i]["ending0"],
            generic_questions.iloc[i]["ending1"],
            generic_questions.iloc[i]["ending2"],
            generic_questions.iloc[i]["ending3"],
        )
        print(
            "Brand options:",
            brand_questions.iloc[i]["ending0"],
            brand_questions.iloc[i]["ending1"],
            brand_questions.iloc[i]["ending2"],
            brand_questions.iloc[i]["ending3"],
        )
