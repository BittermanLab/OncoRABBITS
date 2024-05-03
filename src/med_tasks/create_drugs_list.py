# file name: fetch_drug_info.py

import requests
import pandas as pd
import os
import sys


def fetch_drug_info(max_rxcui):
    data_pairs = []  # Initialize container for all data pairs

    for rxcui in range(1, max_rxcui + 1):
        try:
            # Step 2: Retrieve drug details using the RxCUI
            details_url = (
                f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
            )
            details_response = requests.get(details_url)

            # Check if the response was successful
            if details_response.status_code != 200:
                continue  # Skip this iteration if the request failed

            details_data = details_response.json()

            # Initialize containers for the names
            generic_names = []
            branded_names = []

            for concept_group in details_data.get("allRelatedGroup", {}).get(
                "conceptGroup", []
            ):
                if "conceptProperties" in concept_group:
                    if concept_group["tty"] == "IN":
                        generic_names = [
                            concept["name"]
                            for concept in concept_group["conceptProperties"]
                        ]
                    elif concept_group["tty"] == "BN":
                        branded_names = [
                            concept["name"]
                            for concept in concept_group["conceptProperties"]
                        ]

            # Pair each brand with the generic name and include the RxCUI
            for brand in branded_names:
                for generic in generic_names:
                    data_pairs.append(
                        {"rxcui": rxcui, "generic": generic, "brand": brand}
                    )

        except Exception as e:
            print(f"Error with RxCUI {rxcui}: {e}")
            continue

    # Create DataFrame from pairs
    return pd.DataFrame(data_pairs)


def main(max_rxcui):
    df_drugs = fetch_drug_info(max_rxcui)

    # Save to CSV in the current directory of the script
    current_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(current_dir, "drug_names.csv")
    df_drugs.to_csv(csv_path, index=False)

    print(f"CSV file saved to {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_drug_info.py <max_rxcui>")
        sys.exit(1)

    max_rxcui = int(sys.argv[1])
    main(max_rxcui)
