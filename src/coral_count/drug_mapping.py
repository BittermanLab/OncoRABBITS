import pandas as pd
import random
from typing import Dict, Any, List
import re
import json
import os
from tqdm.auto import tqdm


class DrugMapper:
    def __init__(
        self, brand_to_generic_csv: str, generic_to_brand_csv: str, seed: int = 42
    ):
        """
        Initializes the DrugMapper class with paths to the CSV files and a seed for random operations.
        """
        self.seed = seed
        random.seed(self.seed)  # Set the seed for reproducibility of random choices

        # Load the dataframes
        self.brand_to_generic_df = pd.read_csv(brand_to_generic_csv)
        self.generic_to_brand_df = pd.read_csv(generic_to_brand_csv)

    def load_drug_map(self, reverse_map: bool = False) -> Dict[str, Any]:
        """
        Load the drug map from the CSV files. If reverse_map is True, map generic to brands,
        otherwise map brand to generic.
        """
        if reverse_map:
            # Map generic to randomly chosen brand with fixed seed
            grouped = self.generic_to_brand_df.groupby("generic")["brand"].apply(list)
            drug_map = {
                generic: random.choice(brands) for generic, brands in grouped.items()
            }
        else:
            # Map brand to generic (simple mapping)
            drug_map = pd.Series(
                self.brand_to_generic_df["generic"].values,
                index=self.brand_to_generic_df["brand"],
            ).to_dict()

        return drug_map

    def load_keywords(self, mapping_type: str) -> Dict[str, Any]:
        """
        Load keywords mapping from the CSV files based on the mapping type.
        """
        if mapping_type == "brand_to_generic":
            brand_to_generic = self.brand_to_generic_df.set_index("brand")[
                "generic"
            ].to_dict()
            return brand_to_generic

        elif mapping_type == "generic_to_brand":
            generic_to_brand = (
                self.generic_to_brand_df.set_index("generic")["brand"]
                .groupby(level=0)
                .apply(list)
                .to_dict()
            )
            # Ensure we only process iterable values
            return {
                v: k
                for k, vs in generic_to_brand.items()
                if isinstance(vs, list)
                for v in vs
            }

        else:
            raise ValueError(
                "Invalid mapping type. Use 'brand_to_generic' or 'generic_to_brand'."
            )

    def load_all_keywords_list(self) -> List[str]:
        """
        Load and deduplicate keywords from both brand to generic and generic to brand mappings.
        """
        btog = self.brand_to_generic_df["brand"].tolist()
        gtob = self.generic_to_brand_df["generic"].tolist()

        # Deduplicate keywords
        keywords = list(set(btog + gtob))
        return keywords
