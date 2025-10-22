"""Type coverage analysis for Pokémon teams."""


from src.data.pokedex import Pokemon
from src.data.types import TypeChart


class CoverageAnalyzer:
    """Analyze offensive and defensive type coverage for teams."""

    def __init__(self, type_chart: TypeChart):
        self.type_chart = type_chart

    def offensive_coverage_score(self, team: list[Pokemon]) -> float:
        """
        Calculate offensive type coverage (0-1).

        Score = (number of defending types team can hit super-effectively) / 18

        A defending type is "covered" if at least one team member has STAB
        super-effective against it.
        """
        covered_types = set()

        # For each team member, check which types they hit super-effectively with STAB
        for mon in team:
            for mon_type in mon.types:
                # Check all defending types
                for def_type in self.type_chart.types:
                    effectiveness = self.type_chart.get_effectiveness(mon_type, def_type)
                    if effectiveness > 1.0:  # Super-effective (2x)
                        covered_types.add(def_type)

        return len(covered_types) / 18.0

    def defensive_coverage_score(self, team: list[Pokemon]) -> float:
        """
        Calculate defensive type coverage (0-1).

        For each attacking type, check if team has multiple weaknesses
        and no resists/immunities. Penalize uncovered weaknesses.

        Score = 1 - (penalty / 18)
        """
        penalty = 0

        for atk_type in self.type_chart.types:
            weak_count = 0
            resist_count = 0

            for mon in team:
                effectiveness = self.type_chart.get_matchup(atk_type, mon.types)

                if effectiveness > 1.0:
                    weak_count += 1
                elif effectiveness < 1.0:
                    resist_count += 1

            # Penalize if 2+ weaknesses and 0 resists/immunities
            if weak_count >= 2 and resist_count == 0:
                penalty += 1

        return 1.0 - (penalty / 18.0)

    def type_coverage_score(
        self, team: list[Pokemon], offensive_weight: float = 0.6, defensive_weight: float = 0.4
    ) -> float:
        """
        Calculate combined type coverage score.

        Default weights: 60% offensive, 40% defensive
        """
        offensive = self.offensive_coverage_score(team)
        defensive = self.defensive_coverage_score(team)

        return offensive_weight * offensive + defensive_weight * defensive

    def get_team_weaknesses(self, team: list[Pokemon]) -> dict:
        """Get all attacking types and how many team members are weak to each."""
        weakness_counts = {}

        for atk_type in self.type_chart.types:
            weak_count = sum(
                1
                for mon in team
                if self.type_chart.get_matchup(atk_type, mon.types) > 1.0
            )
            if weak_count > 0:
                weakness_counts[atk_type] = weak_count

        return weakness_counts

    def get_team_resistances(self, team: list[Pokemon]) -> dict:
        """Get all attacking types and how many team members resist each."""
        resistance_counts = {}

        for atk_type in self.type_chart.types:
            resist_count = sum(
                1
                for mon in team
                if self.type_chart.get_matchup(atk_type, mon.types) < 1.0
            )
            if resist_count > 0:
                resistance_counts[atk_type] = resist_count

        return resistance_counts

    def get_weaknesses_covered(
        self, input_team: list[Pokemon], full_team: list[Pokemon]
    ) -> list[str]:
        """
        Get weaknesses that were improved by adding the trio.

        A weakness is "covered" if:
        1. Input team had 2+ weaknesses to that type with 0 resists
        2. Full team has fewer weaknesses OR added resists

        Returns:
            List of type names that were covered/improved
        """
        covered = []
        input_weaknesses = self.get_team_weaknesses(input_team)
        input_resistances = self.get_team_resistances(input_team)
        full_weaknesses = self.get_team_weaknesses(full_team)
        full_resistances = self.get_team_resistances(full_team)

        for atk_type in self.type_chart.types:
            input_weak = input_weaknesses.get(atk_type, 0)
            input_resist = input_resistances.get(atk_type, 0)
            full_weak = full_weaknesses.get(atk_type, 0)
            full_resist = full_resistances.get(atk_type, 0)

            # Was this a problematic weakness? (2+ weaknesses, 0 resists)
            was_problem = input_weak >= 2 and input_resist == 0

            # Did adding the trio help? (added resists OR reduced weaknesses)
            added_help = full_resist > input_resist or full_weak < input_weak

            if was_problem and added_help:
                covered.append(atk_type)

        return covered
