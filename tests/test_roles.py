"""Unit tests for RoleDetector (run from repo root: pytest)."""

from src.data.pokedex import Pokemon
from src.features.roles import FAST_SPEED_THRESHOLD, RoleDetector


def mon(name, speed=50, learnset=()):
    return Pokemon(name=name, types=["Normal"], base_stats={"spe": speed}, learnset=list(learnset))


def test_no_roles_for_slow_mon_without_role_moves():
    assert RoleDetector.detect_roles(mon("Slowpoke", learnset=["Tackle"])) == set()


def test_each_role_detected_from_moves():
    assert RoleDetector.detect_roles(mon("A", learnset=["Stealth Rock"])) == {"hazard_setter"}
    assert RoleDetector.detect_roles(mon("B", learnset=["Defog"])) == {"hazard_removal"}
    assert RoleDetector.detect_roles(mon("C", learnset=["U-turn"])) == {"pivot"}
    assert RoleDetector.detect_roles(mon("D", learnset=["Sucker Punch"])) == {"speed_control"}


def test_speed_threshold_boundary():
    assert "speed_control" not in RoleDetector.detect_roles(mon("Slow", FAST_SPEED_THRESHOLD - 1))
    assert "speed_control" in RoleDetector.detect_roles(mon("Fast", FAST_SPEED_THRESHOLD))


def test_multiple_roles_on_one_mon():
    roles = RoleDetector.detect_roles(
        mon("Multi", speed=120, learnset=["Spikes", "Rapid Spin", "Volt Switch"])
    )
    assert roles == {"hazard_setter", "hazard_removal", "pivot", "speed_control"}


def test_team_role_coverage_unions_roles():
    team = [mon("A", learnset=["Stealth Rock"]), mon("B", learnset=["Flip Turn"])]
    assert RoleDetector.team_role_coverage(team) == {"hazard_setter", "pivot"}
    assert RoleDetector.team_role_coverage([]) == set()


def test_role_diversity_score_scales_and_caps():
    assert RoleDetector.role_diversity_score([]) == 0.0
    one = [mon("A", learnset=["Stealth Rock"])]
    assert RoleDetector.role_diversity_score(one) == 0.25
    two = one + [mon("B", learnset=["Defog"])]
    assert RoleDetector.role_diversity_score(two) == 0.5
    full = [mon("F", speed=130, learnset=["Spikes", "Defog", "U-turn"])]
    assert RoleDetector.role_diversity_score(full) == 1.0  # 1.0 + 0.1 bonus capped at 1.0


def test_get_roles_added_uses_friendly_names():
    base = [mon("A", learnset=["Stealth Rock"])]
    full = base + [mon("B", learnset=["Rapid Spin", "U-turn"])]
    assert sorted(RoleDetector.get_roles_added(base, full)) == ["Hazard Control", "Pivot"]
    assert RoleDetector.get_roles_added(full, full) == []
