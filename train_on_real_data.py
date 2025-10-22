"""
Train ML model on REAL Pokemon Showdown battle outcomes.

Compare feature importances with synthetic data model.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

from src.data.pokedex import Pokedex
from src.data.types import TypeChart
from src.data.usage import UsageStats
from src.features.coverage import CoverageAnalyzer
from src.features.meta import MetaAnalyzer
from src.features.roles import RoleDetector


def extract_features(team_names: list[str], pokedex, coverage_analyzer, meta_analyzer) -> np.ndarray:
    """Extract 7 features from a team (same as synthetic model)."""

    # Convert names to Pokemon objects
    team = []
    for name in team_names:
        mon = pokedex.get(name)
        if mon:
            team.append(mon)

    if len(team) != 6:
        return None

    # Type coverage features
    type_score = coverage_analyzer.type_coverage_score(team)

    # Meta matchup
    meta_score = meta_analyzer.meta_coverage_score(team)

    # Role diversity
    role_detector = RoleDetector()
    roles = set()
    for mon in team:
        mon_roles = role_detector.detect_roles(mon)
        roles.update(mon_roles)
    role_score = len(roles) / 4  # Max 4 roles

    # Secondary features
    avg_speed = np.mean([mon.base_stats['spe'] for mon in team])
    type_diversity = len(set([t for mon in team for t in mon.types]))

    # Physical/special balance
    physical_count = sum(1 for mon in team if mon.base_stats['atk'] > mon.base_stats['spa'])
    balance = min(physical_count, 6 - physical_count) / 3

    # Bulk
    avg_bulk = np.mean([(mon.base_stats['hp'] + mon.base_stats['def'] + mon.base_stats['spd']) / 3 for mon in team])

    return np.array([
        type_score,
        meta_score,
        role_score,
        avg_speed,
        type_diversity,
        balance,
        avg_bulk
    ])


def load_battles(battles_file: Path):
    """Load battles from JSONL file."""
    battles = []
    with open(battles_file) as f:
        for line in f:
            battles.append(json.loads(line))
    return battles


def train_on_real_data():
    """Train model on real battle outcomes."""

    print("Loading Pokemon data...")
    pokedex = Pokedex()
    type_chart = TypeChart()
    usage_stats = UsageStats()

    coverage_analyzer = CoverageAnalyzer(type_chart)
    meta_analyzer = MetaAnalyzer(type_chart, pokedex, usage_stats)

    print("Loading real battles...")
    battles = load_battles(Path("data/replays/battles_fast.jsonl"))
    print(f"  Loaded {len(battles)} battles")

    print("\nExtracting features from teams...")
    X = []
    y = []

    for i, battle in enumerate(battles):
        # Extract features for both teams
        p1_features = extract_features(battle['p1_team'], pokedex, coverage_analyzer, meta_analyzer)
        p2_features = extract_features(battle['p2_team'], pokedex, coverage_analyzer, meta_analyzer)

        if p1_features is None or p2_features is None:
            continue

        # Label: winner = 1, loser = 0
        if battle['winner'] == 'p1':
            X.append(p1_features)
            y.append(1)
            X.append(p2_features)
            y.append(0)
        else:
            X.append(p1_features)
            y.append(0)
            X.append(p2_features)
            y.append(1)

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(battles)} battles...")

    X = np.array(X)
    y = np.array(y)

    print(f"\n✓ Created dataset:")
    print(f"  Total teams: {len(X)}")
    print(f"  Winners: {sum(y)} | Losers: {len(y) - sum(y)}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining on REAL battle data...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    print(f"✓ Training R²: {train_score:.4f}")
    print(f"✓ Validation R²: {val_score:.4f}")

    # Feature importances
    feature_names = [
        'type_score',
        'meta_score',
        'role_score',
        'avg_speed',
        'type_diversity',
        'balance',
        'avg_bulk'
    ]

    importances = model.feature_importances_

    print(f"\n{'='*60}")
    print("REAL DATA: Feature Importances")
    print(f"{'='*60}")

    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        bar = '█' * int(importance * 100)
        print(f"{name:20s}: {importance:6.4f} {bar}")

    print(f"\n{'='*60}")
    print("COMPARISON: Real Data vs Synthetic Data")
    print(f"{'='*60}")

    # Expected importances from synthetic model
    synthetic_importances = {
        'meta_score': 0.5331,
        'role_score': 0.2644,
        'type_score': 0.1705,
        'balance': 0.0116,
        'avg_speed': 0.0092,
        'avg_bulk': 0.0078,
        'type_diversity': 0.0033
    }

    print(f"\n{'Feature':<20} {'Real Data':<12} {'Synthetic':<12} {'Difference':<12}")
    print("-" * 60)

    for name in feature_names:
        real_imp = dict(zip(feature_names, importances))[name]
        synth_imp = synthetic_importances[name]
        diff = real_imp - synth_imp
        sign = '+' if diff > 0 else ''
        print(f"{name:<20} {real_imp:>6.1%}       {synth_imp:>6.1%}       {sign}{diff:>6.1%}")

    # Save model
    output_path = Path("models/real_data_model.pkl")
    output_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\n✓ Model saved to {output_path}")

    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")

    real_meta = dict(zip(feature_names, importances))['meta_score']
    real_type = dict(zip(feature_names, importances))['type_score']
    synth_meta = synthetic_importances['meta_score']
    synth_type = synthetic_importances['type_score']

    if abs(real_meta - synth_meta) > 0.10:
        print(f"⚠️  SIGNIFICANT DIFFERENCE in meta_score importance:")
        print(f"   Real data: {real_meta:.1%} vs Synthetic: {synth_meta:.1%}")
        print(f"   → Real battle outcomes {('prioritize' if real_meta > synth_meta else 'de-prioritize')} meta matchup")
    else:
        print(f"✓ Meta matchup importance SIMILAR between real and synthetic data")
        print(f"   Real: {real_meta:.1%} vs Synthetic: {synth_meta:.1%}")

    if abs(real_type - synth_type) > 0.10:
        print(f"\n⚠️  SIGNIFICANT DIFFERENCE in type_score importance:")
        print(f"   Real data: {real_type:.1%} vs Synthetic: {synth_type:.1%}")
        print(f"   → Real battle outcomes {('prioritize' if real_type > synth_type else 'de-prioritize')} type coverage")
    else:
        print(f"\n✓ Type coverage importance SIMILAR between real and synthetic data")
        print(f"   Real: {real_type:.1%} vs Synthetic: {synth_type:.1%}")

    print(f"\nConclusion:")
    if abs(real_meta - synth_meta) < 0.10 and abs(real_type - synth_type) < 0.10:
        print("  Synthetic data model learned weights that ALIGN with real battle outcomes!")
        print("  → Weak supervision approach was effective")
    else:
        print("  Real battle data reveals DIFFERENT importance weights")
        print("  → Training on real data provides better insights than synthetic approach")


if __name__ == "__main__":
    train_on_real_data()
