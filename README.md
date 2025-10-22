# Pokemon Real Battle Data Experiment

Quick POC to scrape real Pokemon Showdown battles and train ML model on actual outcomes.

## Goal

Compare synthetic data training vs real battle data training to see if learned feature importances differ.

## Setup

```bash
pip install requests
```

## Scraping

```bash
# Test with 50 replays first (~1 minute)
python scrape_replays.py

# Will fetch from gen9ou tier, rating 1400+
# Saves to data/replays/battles.jsonl
```

## Time Estimates

- **50 replays:** ~5-10 minutes (test run)
- **1,000 replays:** ~1-2 hours
- **5,000 replays:** ~5-8 hours (depends on success rate)

Success rate depends on:
- How many battles exist in ID range (404s)
- How many have valid teams (6v6)
- How many meet rating threshold (1400+)
- How many have clear winner (no forfeits)

Typical success rate: 10-30%

## Data Format

JSONL file with one battle per line:
```json
{
  "battle_id": "gen9ou-2140000123",
  "p1_team": ["Garchomp", "Kingambit", "Great Tusk", "Gholdengo", "Dragapult", "Gliscor"],
  "p2_team": ["Landorus-Therian", "Kingambit", "Raging Bolt", "Primarina", "Tinkaton", "Iron Valiant"],
  "winner": "p1",
  "p1_rating": 1650,
  "p2_rating": 1620,
  "rating_diff": 30
}
```

## Results

### Quick POC (100 battles)

✅ **COMPLETED** - Trained model on 100 real battles

**Shocking finding:** Real battle data learned COMPLETELY DIFFERENT weights than synthetic data!

| Feature | Real Battles | Synthetic | Difference |
|---------|-------------|-----------|------------|
| meta_score | **7.4%** | **53.3%** | **-45.9%** ⚠️ |
| role_score | **27.7%** | 26.4% | +1.2% |
| balance | 17.0% | 1.2% | +15.9% |
| bulk | 15.6% | 0.8% | +14.8% |

**Key insight:** Weak supervision injected bias. Our hypothesis (meta=40%) → model learned (meta=53%), but REAL battles show meta only matters 7.4%!

**Problem:** Only 26 teams (severe overfitting). Need more data.

### Scale Up (5,000 battles)

🔄 **IN PROGRESS** - Scraping 5,000 battles (~77 minutes)

This will give us 10,000 labeled teams for reliable training.

## Analysis

See `FINDINGS.md` for detailed analysis of synthetic vs real data comparison.

## Next Steps (After 5K Scrape)

1. ✅ Scrape 5,000 battles (in progress)
2. Train model on 10K labeled teams
3. Validate findings with better statistics
4. If results hold: deploy real-data model to main repo
5. Update resume with honest "I discovered my initial approach was biased" narrative
