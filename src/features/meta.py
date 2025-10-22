"""Meta matchup analysis against top threats."""


from src.data.pokedex import Pokedex, Pokemon
from src.data.types import TypeChart
from src.data.usage import UsageStats


class MetaAnalyzer:
    """Analyze team matchups against meta threats."""

    def __init__(self, type_chart: TypeChart, pokedex: Pokedex, usage_stats: UsageStats):
        self.type_chart = type_chart
        self.pokedex = pokedex
        self.usage_stats = usage_stats

    def has_check(self, team: list[Pokemon], threat: Pokemon) -> bool:
        """
        Check if team has at least one counter/check to a threat.

        Simplified heuristic:
        - Type advantage (resists threat's types OR super-effective against threat)
        - Speed advantage (faster than threat)
        """
        for mon in team:
            # Check type advantage
            # Can mon resist threat's attacks?
            threat_effectiveness = max(
                self.type_chart.get_matchup(threat_type, mon.types)
                for threat_type in threat.types
            )

            # Can mon hit threat super-effectively?
            mon_effectiveness = max(
                self.type_chart.get_matchup(mon_type, threat.types) for mon_type in mon.types
            )

            # Check speed advantage
            speed_advantage = mon.speed > threat.speed

            # Consider it a check if:
            # 1. Resists threat (effectiveness < 1.0) OR
            # 2. Hits super-effectively AND is faster
            if threat_effectiveness < 1.0 or (mon_effectiveness > 1.0 and speed_advantage):
                return True

        return False

    def meta_coverage_score(self, team: list[Pokemon], top_k: int = 15) -> float:
        """
        Calculate meta coverage score (0-1).

        For each of top-K most-used Pokémon, check if team has a counter.
        Weight by usage percentage.

        Score = Σ(usage_pct * has_check) / Σ(usage_pct)
        """
        top_threats = self.usage_stats.get_top_k(k=top_k)

        if not top_threats:
            return 0.0

        total_weighted = 0.0
        total_weight = 0.0

        for entry in top_threats:
            threat = self.pokedex.get(entry.name)
            if threat is None:
                continue  # Skip if not in Pokédex

            has_check = self.has_check(team, threat)
            weight = entry.usage_pct

            total_weighted += weight * (1.0 if has_check else 0.0)
            total_weight += weight

        return total_weighted / total_weight if total_weight > 0 else 0.0

    def get_unchecked_threats(self, team: list[Pokemon], top_k: int = 15) -> list[str]:
        """Get list of meta threats that the team struggles against."""
        top_threats = self.usage_stats.get_top_k(k=top_k)
        unchecked = []

        for entry in top_threats:
            threat = self.pokedex.get(entry.name)
            if threat is None:
                continue

            if not self.has_check(team, threat):
                unchecked.append(entry.name)

        return unchecked

    def get_threats_handled(
        self, input_team: list[Pokemon], full_team: list[Pokemon], top_k: int = 15
    ) -> list[str]:
        """
        Get threats that are now handled after adding the trio.

        Returns:
            List of threat names that input team couldn't check but full team can
        """
        input_unchecked = set(self.get_unchecked_threats(input_team, top_k))
        full_unchecked = set(self.get_unchecked_threats(full_team, top_k))

        # Threats that were unchecked but are now checked
        handled = list(input_unchecked - full_unchecked)

        return handled
