# Validating Weak Supervision: A Case Study in Competitive Pokémon Team Recommendation

**Research Report**

**Author:** Josh Yeung
**Date:** October 2025
**Repository:** [pokemon-real-data-experiment](https://github.com/yeungjosh/pokemon-real-data-experiment)
**Related Project:** [pokemon-team-recommender](https://github.com/yeungjosh/pokemon-team-recommender)

---

## Abstract

We present a case study on the limitations of weak supervision in machine learning, using competitive Pokémon team recommendation as a domain. We initially trained a gradient boosting model on 10,000 synthetic teams labeled with hand-crafted scoring functions, achieving R²=0.64. However, validation against 5,000 real battle outcomes from Pokémon Showdown revealed the synthetic model dramatically overestimated meta matchup importance (53.3% vs actual 14.8%, a 38.6 percentage point error). Real battles prioritized raw statistical attributes (bulk and speed, 62% combined importance) over strategic factors like meta matchup and role diversity. This work demonstrates that weak supervision can systematically inject false beliefs into models, and that validation against ground truth is critical even when synthetic data appears to produce "good" metrics.

**Key Contributions:**
1. Quantitative demonstration of weak supervision bias (38.6% error in feature importance)
2. Validation methodology for synthetic ML models using real-world outcomes
3. Domain insights: stats > strategy in Pokémon battles (ELO 1000-1986)

---

## 1. Introduction

### 1.1 Motivation

Machine learning practitioners often face a fundamental challenge: obtaining labeled training data is expensive. Weak supervision—where labels are generated programmatically rather than from human annotation—offers an attractive solution. However, weak supervision introduces a risk: **the model may learn patterns from our labeling function rather than true underlying relationships**.

This work investigates this risk through a concrete case study: building a recommender system for competitive Pokémon team building.

### 1.2 Problem Statement

**Objective:** Recommend 3 Pokémon to complete a partial team of 3, optimizing for competitive viability.

**Approach:** Train a machine learning model to learn importance weights for three strategic factors:
- **Type coverage:** Offensive and defensive type matchups
- **Meta matchup:** Ability to check popular threats (Kingambit, Garchomp, etc.)
- **Role diversity:** Presence of utility roles (hazards, pivots, priority)

**Challenge:** No labeled dataset of "good" vs "bad" teams exists. How do we train the model?

### 1.3 Research Questions

1. Can weak supervision (synthetic data with hand-crafted labels) produce a viable model?
2. Do feature importances learned from synthetic data reflect real competitive play?
3. What insights can validation against real outcomes provide?

---

## 2. Related Work

### 2.1 Weak Supervision

**Weak supervision** encompasses techniques where labels are generated programmatically rather than by human experts:
- **Programmatic labeling:** Labels derived from heuristic functions
- **Data augmentation:** Synthetic examples with assumed labels
- **Distant supervision:** Labels from external knowledge bases

**Snorkel** (Ratner et al., 2017) popularized weak supervision by combining multiple noisy labeling functions. However, the fundamental risk remains: **if labeling functions are systematically biased, the model will inherit that bias**.

### 2.2 Pokémon Team Building

Competitive Pokémon is a complex strategic domain:
- **18 types** with multiplicative effectiveness (2×, 0.5×, 0×)
- **~900 Pokémon** (this work focuses on Gen 9 OU tier: ~100 viable picks)
- **Meta evolution:** Usage patterns shift monthly based on tournament results

Prior work:
- **Smogon University:** Human expert team analyses (qualitative)
- **Pokémon Showdown:** 200M+ battles logged, but no public ML models
- **Academic:** Limited research on competitive Pokémon AI (mostly battle simulators)

**Gap:** No prior work validates synthetic Pokémon team models against real battle outcomes.

---

## 3. Methodology: Synthetic Data Approach

### 3.1 Synthetic Data Generation

We generated **10,000 synthetic teams** algorithmically:

```python
for _ in range(10000):
    # Sample 6 Pokémon from top 100 usage (weighted by popularity)
    team = sample_from_usage_distribution(n=6)

    # Apply domain constraints
    while not satisfies_constraints(team):
        resample()

    # Constraints:
    # - Role diversity ≥ 2 roles
    # - Type diversity (no more than 2 of same type)
    # - Usage weighted (realistic team compositions)
```

### 3.2 Weak Supervision: Labeling Function

**Key decision:** We hand-crafted a scoring function based on competitive Pokémon knowledge:

```python
def score_team(team):
    type_score = type_coverage_score(team)      # 0-1
    meta_score = meta_matchup_score(team)       # 0-1
    role_score = role_diversity_score(team)     # 0-1

    # Initial hypothesis: meta and type equally important
    return 0.4 * type_score + 0.4 * meta_score + 0.2 * role_score
```

**Rationale:**
- Meta matchup matters (need to beat Kingambit, Garchomp)
- Type coverage matters (offensive pressure)
- Roles matter less (nice-to-have utility)

**Critical assumption:** These weights reflect true importance in battles.

### 3.3 Feature Engineering

We extracted **7 numerical features** from each team:

| Feature | Description | Hypothesis |
|---------|-------------|------------|
| `type_score` | Offensive + defensive type coverage | Important |
| `meta_score` | % of top 15 meta threats checked | Very important |
| `role_score` | Count of utility roles present / 4 | Moderately important |
| `avg_speed` | Mean base speed stat | Minor importance |
| `avg_bulk` | Mean (HP + DEF + SPD) / 3 | Minor importance |
| `type_diversity` | Count of unique types | Minor importance |
| `balance` | Physical/special attacker balance | Minor importance |

### 3.4 Model Training

**Model:** sklearn `GradientBoostingRegressor`
- 100 trees (n_estimators)
- Max depth 4 (prevent overfitting)
- Learning rate 0.1

**Dataset split:**
- Training: 8,000 teams
- Validation: 2,000 teams

**Training procedure:**
```python
X = extract_features(teams)  # Shape: (10000, 7)
y = score_team(teams)        # Shape: (10000,) - our synthetic labels

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)
```

---

## 4. Results: Synthetic Model

### 4.1 Model Performance

**Metrics:**
- Training R²: 0.6850
- Validation R²: 0.6421
- **Train/val gap:** 4.3% (minimal overfitting)

**Interpretation:** The model achieved "good" R² and generalized well to held-out synthetic data.

### 4.2 Learned Feature Importances

The model learned the following importance weights:

| Feature | Importance | Our Hypothesis |
|---------|------------|----------------|
| **meta_score** | **53.3%** | 40% (hypothesis) |
| role_score | 26.4% | 20% (hypothesis) |
| type_score | 17.1% | 40% (hypothesis) |
| balance | 1.2% | Minor |
| avg_speed | 0.9% | Minor |
| avg_bulk | 0.8% | Minor |
| type_diversity | 0.3% | Minor |

### 4.3 Initial Interpretation

**Surprising finding:** Meta matchup learned as 53.3% (vs our 40% hypothesis).

**Our conclusion at the time:**
> "The model discovered that meta matchup matters **3× more** than type coverage. This aligns with competitive intuition - you face Kingambit in 40%+ of battles, so countering it dominates team building."

**This conclusion was WRONG.** (See Section 6)

---

## 5. Validation Study: Real Battle Data

### 5.1 Motivation

To validate the synthetic model, we collected real battle outcomes from Pokémon Showdown.

**Hypothesis to test:** Do learned feature importances reflect actual predictors of wins?

### 5.2 Data Collection

**Source:** [Pokémon Showdown](https://pokemonshowdown.com/) public replays

**Method:** Web scraping using search API
```
https://replay.pokemonshowdown.com/search.json?format=gen9ou&page=N
```

**Dataset:**
- 5,000 battles scraped (99.1% success rate)
- Tier: Gen 9 OU (Over Used)
- Date range: October 2025
- Rating range: 1000-1986 ELO

**Filtering:**
- Both teams must have 6 Pokémon
- Clear winner (no forfeits/disconnects)
- Both players rated ≥1000 ELO
- All Pokémon in our 100-mon Pokédex

**Final dataset:**
- Usable teams: 996 out of 10,000 potential (10%)
- Training: 796 teams
- Validation: 200 teams

**Why only 10% usable?**
Our Pokédex contained only 100 top-tier Pokémon. Real battles use ~300+ different Pokémon. Teams with Pokemon outside our dataset were excluded.

### 5.3 Labeling: Real Outcomes

**Ground truth labels:**
```python
for battle in battles:
    winner_team = battle['winner_team']
    loser_team = battle['loser_team']

    X.append(extract_features(winner_team))
    y.append(1)  # Winner

    X.append(extract_features(loser_team))
    y.append(0)  # Loser
```

**No hand-crafted scoring function.** Labels come from actual battle results.

### 5.4 Model Training on Real Data

Same model architecture as synthetic approach:
```python
model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_real, y_real)  # y_real = actual wins/losses
```

---

## 6. Results: Comparison

### 6.1 Model Performance

**Real data model:**
- Training R²: 0.4288
- Validation R²: **-0.1354** (NEGATIVE!)

**Interpretation:**
Predicting battle outcomes from team composition alone is **fundamentally difficult**. Confounding factors dominate:
- Player skill (ELO 1000-1986 variance)
- Battle RNG (critical hits, move misses)
- Move/EV/IV variations not captured

**However:** Feature importances still reveal true priorities.

### 6.2 Feature Importances: Synthetic vs Real

| Feature | Synthetic | Real Data | Difference |
|---------|-----------|-----------|------------|
| **meta_score** | **53.3%** | **14.8%** | **-38.6%** ⚠️ |
| **avg_bulk** | 0.8% | **31.4%** | **+30.6%** ⚠️ |
| **avg_speed** | 0.9% | **30.9%** | **+29.9%** ⚠️ |
| role_score | 26.4% | 2.3% | -24.1% ⚠️ |
| type_score | 17.1% | 9.7% | -7.4% ✅ |
| type_diversity | 0.3% | 7.3% | +7.0% |
| balance | 1.2% | 3.7% | +2.5% |

### 6.3 Key Findings

**1. Meta matchup DRASTICALLY overrated**
- Synthetic: 53.3% (claimed most important)
- Real: 14.8% (middle-tier importance)
- Error: -38.6 percentage points

**2. Raw stats DRASTICALLY underrated**
- Bulk: 0.8% → 31.4% (+30.6%)
- Speed: 0.9% → 30.9% (+29.9%)
- Combined: **62.3% of importance**

**3. Role diversity overrated**
- Synthetic: 26.4%
- Real: 2.3%
- Error: -24.1 percentage points

### 6.4 Domain Interpretation

**What real battles taught us:**

✅ **Stats > Strategy**
High-bulk, high-speed Pokémon win more than strategically "balanced" teams. Raw power matters.

✅ **Meta matchup is overrated**
Having a Kingambit check matters less than we thought. Stats compensate for poor matchups.

✅ **Utility roles matter least**
Having hazards/pivots/priority is nice-to-have, not game-deciding.

**Why our synthetic model failed:**
We over-indexed on strategic complexity and undervalued stat totals. The competitive community's narrative ("meta matchup is everything") misled our labeling function.

---

## 7. Discussion

### 7.1 Weak Supervision Can Systematically Mislead

**The core problem:** Our labeling function encoded false beliefs:
```python
# We believed:
score = 0.4 * meta + 0.4 * type + 0.2 * role

# Reality:
score ≈ 0.31 * bulk + 0.31 * speed + 0.15 * meta + ...
```

**The model "discovered" meta=53%** by fitting noise and non-linearities in our synthetic labels. This felt like a breakthrough ("ML found meta matters 3× more!"), but it was **circular reasoning** - the model learned from our biased labels.

### 7.2 Synthetic Metrics Can Be Misleading

**Synthetic model:** R²=0.64 (good!)
**Real model:** R²=-0.14 (terrible!)

**Lesson:** High R² on synthetic data ≠ validity. The model "worked" by learning patterns in our made-up scoring function, not true relationships.

### 7.3 When to Trust Weak Supervision

**Weak supervision can work when:**
1. Labeling functions based on ground truth proxies (not beliefs)
2. Multiple independent labeling functions (Snorkel approach)
3. Validation against real outcomes performed early

**Our mistake:** Single labeling function based on competitive narrative, no real-world validation until after deployment.

### 7.4 Limitations of This Study

**1. Small usable dataset**
- Only 996/10,000 teams (10%) due to Pokédex filtering
- Limits statistical power

**2. Rating confounding**
- ELO 1000-1986 range (high variance)
- Player skill >> team quality at low ELO

**3. Missing context**
- Moves, EVs, IVs not captured
- Team preview strategy not modeled

**4. Top-tier bias**
- Only analyzing teams with 100 most-used Pokémon
- May not generalize to off-meta teams

**5. Observational data**
- Cannot establish causality
- Correlation between stats and wins may reflect player skill (good players use high-stat mons)

---

## 8. Conclusions

### 8.1 Summary

We demonstrated that weak supervision can inject **systematic bias** into machine learning models. Our synthetic Pokémon team recommender achieved R²=0.64 but learned feature importances contradicted by real battle outcomes (38.6% error for meta matchup importance).

### 8.2 Practical Takeaways

**For ML practitioners:**

1. **Validate early:** Don't wait until deployment to test against real outcomes
2. **Question your labels:** If using weak supervision, explicitly state assumptions and test them
3. **Distrust "surprising discoveries":** If the model learns something unexpected from synthetic data, it's probably fitting your biases
4. **Metrics aren't everything:** High R² on synthetic data doesn't validate your approach

**For Pokémon team building:**

1. **Stats > Strategy:** Bulk and speed matter more than meta matchups
2. **Meta matchup overrated:** Community narrative may mislead (confirmation bias)
3. **Complexity is high:** R²=-0.14 shows team composition alone doesn't predict wins

### 8.3 Future Work

**1. Expand Pokédex**
- Include 300+ Pokémon to avoid filtering bias
- Analyze off-meta team effectiveness

**2. Control for player skill**
- Filter for high ELO only (1600+)
- Matched pairs design (same player, different teams)

**3. Include move data**
- Model move choices (not just Pokémon selection)
- Incorporate EVs/IVs

**4. Causal inference**
- Instrument variables to isolate team quality from player skill
- Natural experiments (meta shifts after bans)

**5. Alternative objectives**
- Instead of "predict wins," optimize for "team diversity"
- Recommend anti-meta teams (unexplored strategies)

---

## 9. References

### Academic
- Ratner, A., et al. (2017). "Snorkel: Rapid Training Data Creation with Weak Supervision." VLDB.
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD.
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." Annals of Statistics.

### Domain
- Smogon University. "Gen 9 OU Metagame Discussion." https://www.smogon.com/forums/
- Pokémon Showdown. "Usage Statistics." https://www.smogon.com/stats/
- Pokémon Database. "Pokémon Type Chart." https://pokemondb.net/

### Software
- scikit-learn: Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." JMLR.
- Gradio: Abid, A., et al. (2019). "Gradio: Hassle-Free Sharing and Testing of ML Models." arXiv.

---

## Appendix A: Reproducibility

### Code Availability

**Main project:**
https://github.com/yeungjosh/pokemon-team-recommender

**Validation study:**
https://github.com/yeungjosh/pokemon-real-data-experiment

### Data Availability

**Pokémon data:**
Included in repositories (100 Pokémon, types, moves, usage stats)

**Battle replays:**
5,000 replays scraped from Pokémon Showdown (publicly accessible via their API). Raw data: `data/replays/battles_fast.jsonl` (~5MB, not committed to git).

**Reproduction steps:**
```bash
# Clone validation repo
git clone https://github.com/yeungjosh/pokemon-real-data-experiment
cd pokemon-real-data-experiment

# Install dependencies
pip install -r requirements.txt

# Scrape new battles (optional - can use existing data)
python scrape_fast.py

# Train model on real data
python train_on_real_data.py
```

Expected output: Feature importance comparison table (takes ~10 seconds).

### Computing Environment

- **OS:** macOS Sonoma 14.3
- **Python:** 3.12
- **Key packages:** scikit-learn 1.3.0, pandas 2.0.3, numpy 1.24.3
- **Hardware:** M1 MacBook Pro (training takes <10 seconds)

---

## Appendix B: Interview Q&A

**Q: "Why did you choose Pokémon for this study?"**

A: "I needed a domain where (1) synthetic data generation is feasible, (2) ground truth labels exist (battle outcomes), and (3) I had domain expertise to craft plausible labeling functions. Pokémon fit perfectly - I could generate reasonable-looking teams, but I needed real battles to validate my assumptions."

---

**Q: "Your real-data model has R²=-0.14. Doesn't that mean it's useless?"**

A: "For predicting individual battle outcomes, yes. But feature importances still revealed that my synthetic model had the wrong priorities. Even a 'bad' model can teach you your assumptions were wrong, which is valuable. In production, I'd focus on diversifying recommendations rather than predicting wins."

---

**Q: "What would you do differently if starting over?"**

A: "I'd validate against real outcomes before investing in the full pipeline. A quick POC (100 battles) would've shown the mismatch immediately. I'd also use multiple independent labeling functions (Snorkel-style) rather than a single hand-crafted one, reducing bias."

---

**Q: "How would you improve the real-data model?"**

A: "Three directions: (1) Expand Pokédex to 300+ Pokémon (currently only 10% of teams usable), (2) Filter for high ELO only (1600+) to reduce skill confounding, and (3) Include move/EV/IV data. But honestly, predicting wins from teams alone may be the wrong objective - recommending diverse/anti-meta teams is more valuable."

---

**END OF REPORT**
