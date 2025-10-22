"""Pokémon data loading and utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Pokemon:
    """Pokémon species data."""

    name: str
    types: List[str]
    base_stats: Dict[str, int]
    learnset: List[str]
    sprite: str = ""

    @property
    def speed(self) -> int:
        return self.base_stats["spe"]

    def can_learn(self, move: str) -> bool:
        """Check if this Pokémon can learn a specific move."""
        return move in self.learnset


class Pokedex:
    """Load and query Pokémon species data."""

    def __init__(self, data_path: Path = None):
        if data_path is None:
            data_path = Path(__file__).parents[2] / "data" / "raw" / "pokedex.json"

        with open(data_path) as f:
            raw_data = json.load(f)

        self.pokemon: Dict[str, Pokemon] = {}
        for entry in raw_data:
            mon = Pokemon(
                name=entry["name"],
                types=entry["types"],
                base_stats=entry["baseStats"],
                learnset=entry["learnset"],
                sprite=entry.get("sprite", ""),
            )
            self.pokemon[mon.name] = mon

    def get(self, name: str) -> Pokemon:
        """Get Pokémon by name."""
        return self.pokemon.get(name)

    def exists(self, name: str) -> bool:
        """Check if a Pokémon exists in the dex."""
        return name in self.pokemon

    def all_names(self) -> List[str]:
        """Get list of all Pokémon names."""
        return list(self.pokemon.keys())

    def filter_by_type(self, type_name: str) -> List[Pokemon]:
        """Get all Pokémon that have a specific type."""
        return [mon for mon in self.pokemon.values() if type_name in mon.types]

    def filter_by_move(self, move: str) -> List[Pokemon]:
        """Get all Pokémon that can learn a specific move."""
        return [mon for mon in self.pokemon.values() if mon.can_learn(move)]
