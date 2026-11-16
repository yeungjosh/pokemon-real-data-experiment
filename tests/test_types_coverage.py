"""Unit tests for TypeChart and CoverageAnalyzer (run from repo root: pytest)."""

import pytest

from src.data.pokedex import Pokemon
from src.data.types import TypeChart
from src.features.coverage import CoverageAnalyzer


def mon(name, *types):
    return Pokemon(name=name, types=list(types), base_stats={"spe": 50}, learnset=[])


@pytest.fixture(scope="module")
def chart():
    return TypeChart()


@pytest.fixture(scope="module")
def analyzer(chart):
    return CoverageAnalyzer(chart)


def test_chart_has_18_types_and_square_matrix(chart):
    assert len(chart.types) == 18
    for atk in chart.types:
        assert set(chart.chart[atk]) == set(chart.types)


def test_basic_effectiveness(chart):
    assert chart.get_effectiveness("Fire", "Grass") == 2.0
    assert chart.get_effectiveness("Fire", "Water") == 0.5
    assert chart.get_effectiveness("Normal", "Ghost") == 0.0


def test_dual_type_multiplies(chart):
    assert chart.get_matchup("Ice", ["Dragon", "Flying"]) == 4.0
    assert chart.get_matchup("Ground", ["Flying", "Steel"]) == 0.0


def test_weakness_resistance_immunity_lists(chart):
    assert "Ground" in chart.get_weaknesses(["Fire"])
    assert "Fire" in chart.get_resistances(["Fire"])
    assert chart.get_immunities(["Ghost"]).count("Normal") == 1
    assert "Normal" not in chart.get_weaknesses(["Ghost"])


def test_scores_are_in_unit_interval(analyzer):
    team = [mon("a", "Fire"), mon("b", "Water"), mon("c", "Grass", "Poison")]
    for score in (
        analyzer.offensive_coverage_score(team),
        analyzer.defensive_coverage_score(team),
        analyzer.type_coverage_score(team),
    ):
        assert 0.0 <= score <= 1.0


def test_empty_team_scores(analyzer):
    assert analyzer.offensive_coverage_score([]) == 0.0
    assert analyzer.defensive_coverage_score([]) == 1.0


def test_stab_coverage_is_monotonic(analyzer):
    one = [mon("a", "Fire")]
    two = one + [mon("b", "Water")]
    assert analyzer.offensive_coverage_score(two) >= analyzer.offensive_coverage_score(one)


def test_shared_weakness_is_penalized(analyzer):
    # Three Grass types are all weak to Fire and none resists it
    stacked = [mon("a", "Grass"), mon("b", "Grass"), mon("c", "Grass")]
    assert analyzer.defensive_coverage_score(stacked) < 1.0


def test_team_weakness_and_resistance_counts(analyzer):
    team = [mon("a", "Fire"), mon("b", "Fire")]
    assert analyzer.get_team_weaknesses(team)["Water"] == 2
    assert analyzer.get_team_resistances(team)["Fire"] == 2


def test_weaknesses_covered_by_added_resist(analyzer):
    base = [mon("a", "Grass"), mon("b", "Grass")]
    full = base + [mon("c", "Fire")]
    assert "Fire" not in analyzer.get_weaknesses_covered(base, base)
    # Fire was a 2-weak/0-resist problem for base; Fire type adds a resist
    assert "Fire" in analyzer.get_weaknesses_covered(base, full)
