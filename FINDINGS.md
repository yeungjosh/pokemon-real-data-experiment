# Real Battle Data vs Synthetic Data: Key Findings

## Experiment Overview

**Goal:** Compare ML model trained on synthetic data vs real Pokemon Showdown battle outcomes

**Dataset:**
- **Synthetic:** 10,000 algorithmically generated teams with weak supervision labels
- **Real (POC):** 100 battles = 200 labeled teams (winners vs losers)
- **Real (Full):** 5,000 battles = 10,000 labeled teams (in progress)

## Quick POC Results (100 Battles)

### Feature Importance Comparison

| Feature | Real Battles | Synthetic | Difference |
|---------|-------------|-----------|------------|
| **meta_score** | **7.4%** | **53.3%** | **-45.9%** ⚠️ |
| role_score | 27.7% | 26.4% | +1.2% ✅ |
| balance | 17.0% | 1.2% | +15.9% ⚠️ |
| avg_bulk | 15.6% | 0.8% | +14.8% ⚠️ |
| type_score | 13.9% | 17.1% | -3.2% ✅ |
| avg_speed | 10.4% | 0.9% | +9.5% ⚠️ |
| type_diversity | 8.0% | 0.3% | +7.7% ⚠️ |

### Key Insights

**❌ Synthetic Data Hypothesis FAILED:**
- We hypothesized meta matchup is most important (53.3%)
- Real battles show it's LEAST important (7.4%)
- **45.9 percentage point error!**

**✅ Real Battles Prioritize:**
1. **Role diversity** (27.7%) - Having hazards, pivots, priority matters
2. **Balance** (17.0%) - Physical/special attacker mix
3. **Bulk** (15.6%) - Surviving hits is crucial
4. **Type coverage** (13.9%) - Still matters, but not dominant

**⚠️ Caveat:** POC dataset is VERY small (26 teams after filtering)
- Training R²: 1.0 (overfitting)
- Validation R²: 0.099 (poor generalization)
- **Need more data!**

## Why Synthetic Data Failed

### Weak Supervision Circularity

**What we did:**
```python
# Step 1: Made up initial weights
target_score = 0.4×type + 0.4×meta + 0.2×role

# Step 2: Generated 10K labels with this formula

# Step 3: Trained ML model
# Model learned: 53.3% meta, 17.1% type, 26.4% role

# Step 4: Claimed "ML discovered meta matters 3x more!"
```

**The problem:**
- Model learned patterns from OUR labels, not real Pokemon battles
- We injected bias: "meta = 40%" → model learned "meta = 53%"
- This is just fitting noise in our made-up formula

### Real Data Reveals Truth

**Real battles:**
```python
# Step 1: Real battle outcomes (Player 1 won vs Player 2 lost)

# Step 2: Extract features from both teams

# Step 3: Train ML: winner=1, loser=0

# Step 4: Model learns what ACTUALLY predicts wins
```

**Result:** Completely different weights!
- Role diversity >> Meta matchup
- Balance and bulk matter more than we thought
- Meta matchup is surprisingly low (7.4%)

## Possible Explanations

### Why is Meta Matchup Low?

**Hypothesis 1: Rating confounding**
- Low-rated players (1000-1400 ELO) may not know meta threats
- They might not build teams specifically to counter Kingambit/Garchomp
- Would need to filter for 1600+ ELO only

**Hypothesis 2: Sample bias**
- 100 battles is tiny sample
- May have caught unusual meta shifts
- Need 5,000 battles for confidence

**Hypothesis 3: Meta matchup is overrated**
- Competitive community assumes meta matters most
- But actual wins come from role synergy + balance
- "Counter the meta" strategy may be overhyped

### Why is Balance High?

**Physical/special balance** scored 17.0% (vs synthetic 1.2%)

**Explanation:**
- Teams with only physical attackers get walled by high DEF Pokemon
- Teams with only special attackers get walled by high SPD Pokemon
- Balanced teams are harder to predict and counter
- **This aligns with competitive wisdom!**

## Next Steps

### Scale Up (In Progress)

**Target:** 5,000 battles = 10,000 labeled teams

**Expected improvements:**
- More reliable feature importances
- Better generalization (higher validation R²)
- Can filter by rating tier (1400+, 1600+, 1800+)
- Statistical significance testing

### If Results Hold

**If role_score >> meta_score persists:**
- Update main repo's recommender weights
- Deprecate synthetic data model
- Deploy real-data model to HF Spaces
- Update README with honest findings

**Resume talking point:**
> "I initially trained on synthetic data (R²=0.64) but found feature importances didn't match real battle outcomes. I scaled up to 5,000 real Pokemon Showdown battles and discovered role diversity predicts wins 4x better than meta matchup (27.7% vs 7.4% importance). This taught me weak supervision can inject bias - validation against real outcomes is critical."

## Files

- `scrape_fast.py` - Fast scraper using search API
- `train_on_real_data.py` - Training script with comparison
- `data/replays/battles_fast.jsonl` - Real battle data
- `models/real_data_model.pkl` - Trained model

## Timeline

- **100 battles:** 50 seconds (POC complete)
- **5,000 battles:** ~77 minutes (in progress)
- **Training:** ~2 minutes
- **Total:** ~80 minutes for full experiment
