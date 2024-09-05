# irAE Dataset Generation

This README provides an overview of the key files and processes involved in generating the immune-related Adverse Events (irAE) dataset.

## Key Files and Locations

- **Drug Lists**:
  - `data/drug_names/brand_to_generic_df.csv`: Mapping of brand names to generic names
  - `data/drug_names/generic_to_brand_df.csv`: Mapping of generic names to brand names

- **Input Data**:
  - `src/irAE/oncqa.csv`: Original OncQA dataset
  - `src/irAE/immunotherapy_list.csv`: List of immunotherapy drugs and regimens

- **Output Files**:
  - `src/irAE/filtered_oncqa.csv`: OncQA data filtered for immunotherapy drugs
  - `src/irAE/immunotherapy_with_mapped_drugs.csv`: Immunotherapy data with mapped drug names
  - `src/irAE/filtered_drug_counts.csv`: Counts of drugs in the filtered dataset
  - `src/irAE/merged_df.csv`: Combined dataset of filtered OncQA and immunotherapy cases
  - `src/irAE/merged_df_brand_only.csv`: Version of the merged dataset using only brand names
  - `src/irAE/merged_df_generic_only.csv`: Version of the merged dataset using only generic names
  - `src/irAE/generated_prompts.csv`: Generated prompts for the original merged dataset
  - `src/irAE/generated_prompts_brand_only.csv`: Generated prompts for the brand-only dataset
  - `src/irAE/generated_prompts_generic_only.csv`: Generated prompts for the generic-only dataset

## Script Overview

The script performs the following main tasks:

1. Processes the OncQA dataset, extracting relevant text and identifying drug keywords.
2. Processes the immunotherapy dataset, mapping drug names to their brand and generic equivalents.
3. Filters the OncQA dataset to include only cases mentioning immunotherapy drugs.
4. Combines the filtered OncQA data with the immunotherapy data.
5. Creates brand-only and generic-only versions of the merged dataset.
6. Generates prompts for irAE detection and differential diagnosis tasks using the merged datasets.

## Usage

To view the drugs or cases:
- For the list of drugs, check the input drug list files in the `data/drug_names/` directory.
- For the OncQA cases, refer to `src/irAE/filtered_oncqa.csv`.
- For the immunotherapy cases, see `src/irAE/immunotherapy_with_mapped_drugs.csv`.
- For the combined dataset of cases, look at `src/irAE/merged_df.csv`.

To see the generated prompts for different scenarios, check the respective `generated_prompts*.csv` files in the `src/irAE/` directory.