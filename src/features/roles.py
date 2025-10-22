"""Role detection for Pokémon based on learnsets and stats."""


from src.data.pokedex import Pokemon

# Role-defining moves
HAZARD_MOVES = {"Stealth Rock", "Spikes", "Toxic Spikes"}
REMOVAL_MOVES = {"Rapid Spin", "Defog"}
PIVOT_MOVES = {"U-turn", "Volt Switch", "Flip Turn"}
PRIORITY_MOVES = {
    "Extreme Speed",
    "Aqua Jet",
    "Mach Punch",
    "Sucker Punch",
    "Ice Shard",
    "Thunderclap",
}

# Speed tiers
FAST_SPEED_THRESHOLD = 100  # Base speed >= 100 is considered "fast"


class RoleDetector:
    """Detect competitive roles for Pokémon."""

    @staticmethod
    def detect_roles(pokemon: Pokemon) -> set[str]:
        """
        Detect all roles a Pokémon can fill.

        Returns:
            Set of role names: "hazard_setter", "hazard_removal", "pivot", "speed_control"
        """
        roles = set()

        # Check for hazard setter
        if any(move in pokemon.learnset for move in HAZARD_MOVES):
            roles.add("hazard_setter")

        # Check for hazard removal
        if any(move in pokemon.learnset for move in REMOVAL_MOVES):
            roles.add("hazard_removal")

        # Check for pivot
        if any(move in pokemon.learnset for move in PIVOT_MOVES):
            roles.add("pivot")

        # Check for speed control
        # Fast Pokémon OR priority users
        if pokemon.speed >= FAST_SPEED_THRESHOLD or any(
            move in pokemon.learnset for move in PRIORITY_MOVES
        ):
            roles.add("speed_control")

        return roles

    @staticmethod
    def team_role_coverage(team: list[Pokemon]) -> set[str]:
        """Get all roles covered by a team."""
        all_roles = set()
        for mon in team:
            all_roles.update(RoleDetector.detect_roles(mon))
        return all_roles

    @staticmethod
    def role_diversity_score(team: list[Pokemon]) -> float:
        """
        Calculate role diversity score (0-1).

        Target roles: hazard_setter, hazard_removal, pivot, speed_control
        Score = (number of roles present) / 4
        Bonus: +0.1 if all 4 roles covered
        """
        roles = RoleDetector.team_role_coverage(team)
        base_score = len(roles) / 4.0

        # Bonus for complete coverage
        if len(roles) == 4:
            return min(1.0, base_score + 0.1)

        return base_score

    @staticmethod
    def get_roles_added(input_team: list[Pokemon], full_team: list[Pokemon]) -> list[str]:
        """
        Get roles that were added by the trio.

        Returns:
            List of role names that input team didn't have but full team has
        """
        # Map internal role names to user-friendly names
        role_name_map = {
            "hazard_setter": "Hazard Setter",
            "hazard_removal": "Hazard Control",
            "pivot": "Pivot",
            "speed_control": "Speed Control",
        }

        input_roles = RoleDetector.team_role_coverage(input_team)
        full_roles = RoleDetector.team_role_coverage(full_team)

        # Roles that were added
        added_roles = full_roles - input_roles

        # Convert to user-friendly names
        return [role_name_map.get(role, role) for role in added_roles]
