"""Usage statistics loading and utilities."""

import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class UsageEntry:
    """Usage stats for a single Pokémon."""

    name: str
    usage_pct: float
    tier: str
    generation: int
    month: str


class UsageStats:
    """Load and query competitive usage statistics."""

    def __init__(self, data_path: Path = None):
        if data_path is None:
            data_path = Path(__file__).parents[2] / "data" / "raw" / "usage_ou.csv"

        self.df = pd.read_csv(data_path)

    def get_usage(self, name: str) -> float:
        """Get usage percentage for a Pokémon (0-100 range)."""
        match = self.df[self.df["name"] == name]
        if match.empty:
            return 0.0
        return match.iloc[0]["usage_pct"]

    def get_top_k(self, k: int = 15, tier: str = "OU") -> List[UsageEntry]:
        """
        Get top K most-used Pokémon in a tier.

        Args:
            k: Number of Pokémon to return
            tier: Competitive tier (default: OU)

        Returns:
            List of UsageEntry objects, sorted by usage descending
        """
        tier_df = self.df[self.df["tier"] == tier]
        top_k = tier_df.nlargest(k, "usage_pct")

        return [
            UsageEntry(
                name=row["name"],
                usage_pct=row["usage_pct"],
                tier=row["tier"],
                generation=row["generation"],
                month=row["month"],
            )
            for _, row in top_k.iterrows()
        ]

    def get_all_names(self, tier: str = "OU") -> List[str]:
        """Get all Pokémon names in a tier, sorted by usage."""
        tier_df = self.df[self.df["tier"] == tier]
        return tier_df.sort_values("usage_pct", ascending=False)["name"].tolist()
