"""Type effectiveness data loading and utilities."""

import json
from pathlib import Path
from typing import Dict


class TypeChart:
    """Load and query Pokémon type effectiveness data."""

    def __init__(self, data_path: Path = None):
        if data_path is None:
            data_path = Path(__file__).parents[2] / "data" / "raw" / "type_chart.json"

        with open(data_path) as f:
            self.chart: Dict[str, Dict[str, float]] = json.load(f)

        self.types = list(self.chart.keys())

    def get_effectiveness(self, attacking_type: str, defending_type: str) -> float:
        """
        Get type effectiveness multiplier.

        Args:
            attacking_type: The attacking move's type
            defending_type: The defending Pokémon's type

        Returns:
            Effectiveness multiplier (0, 0.5, 1, or 2)
        """
        return self.chart[attacking_type][defending_type]

    def get_matchup(self, attacking_type: str, defending_types: list[str]) -> float:
        """
        Calculate effectiveness against a Pokémon with 1-2 types.

        Args:
            attacking_type: The attacking move's type
            defending_types: List of defending types (1 or 2 elements)

        Returns:
            Combined effectiveness (e.g., 0.25, 0.5, 1, 2, 4)
        """
        multiplier = 1.0
        for def_type in defending_types:
            multiplier *= self.get_effectiveness(attacking_type, def_type)
        return multiplier

    def get_defensive_matchups(self, defending_types: list[str]) -> Dict[str, float]:
        """
        Get all offensive type matchups against a given defender.

        Args:
            defending_types: List of defending types (1 or 2 elements)

        Returns:
            Dict mapping attacking type to effectiveness multiplier
        """
        return {
            atk_type: self.get_matchup(atk_type, defending_types)
            for atk_type in self.types
        }

    def get_weaknesses(self, defending_types: list[str]) -> list[str]:
        """Get types that are super-effective against the defender."""
        matchups = self.get_defensive_matchups(defending_types)
        return [t for t, eff in matchups.items() if eff > 1.0]

    def get_resistances(self, defending_types: list[str]) -> list[str]:
        """Get types that are not very effective against the defender."""
        matchups = self.get_defensive_matchups(defending_types)
        return [t for t, eff in matchups.items() if eff < 1.0]

    def get_immunities(self, defending_types: list[str]) -> list[str]:
        """Get types that have no effect against the defender."""
        matchups = self.get_defensive_matchups(defending_types)
        return [t for t, eff in matchups.items() if eff == 0.0]
