import pandas as pd
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

# Example list of drug names and their replacements
drug_data = {
    "old": ["Aspirin", "Ibuprofen", "Acetaminophen", "Amoxicillin", "year-old"],
    "new": ["Dog", "Cat", "Bird", "Fish", "Chicken"],
}

# Convert dictionary to DataFrame
df_drugs = pd.DataFrame(drug_data)

# Save to CSV in the current directory of the script
csv_path = os.path.join(current_dir, "drug_names.csv")
df_drugs.to_csv(csv_path, index=False)

print(f"CSV file saved to {csv_path}")
