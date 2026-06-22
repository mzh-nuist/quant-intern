# A-Share Multi-Market-Cap Factor Research

Systematic factor research across A-share market-cap layers, from micro-cap (RMB 1.3B) to mega-cap (RMB 2.7T). Built during a 2026 quant internship.

**Key finding**: Factor direction flips across market-cap layers. Textbook signals (low PB = safety margin, high ROE = downside protection) reverse in the micro-cap universe. Directional consistency over 6 years matters more than t-statistic magnitude.

## Project Structure

```
├── research/           # 9 research modules (Q1–Q9, Jupyter notebooks)
│   ├── Q1–Q4           # Micro-cap: drivers, valuations, resilience, style rotation
│   ├── Q5               # CSI 300 large-cap factor validation
│   ├── Q7               # ML-based profitable micro-cap discovery (XGBoost + SHAP + UMAP)
│   ├── Q8               # Factor attribution across market-cap layers
│   ├── Q9a-c            # Interaction detection, feature engineering, time-varying weights
│   └── q7_fixed.py      # Corrected version: regularization + hold-out validation
├── joinquant/          # Strategy implementations on JoinQuant (聚宽) platform
├── images/             # Key visualizations
├── alpha_meaning.md    # Essay: "Alpha is not prediction, it's the price of constraints"
├── 项目复盘_错误与成长.md  # Post-mortem: methodology failures and lessons learned
├── Q1-Q6_面试总结.md    # Interview-ready research summary
└── 策略分析报告.md       # Strategy analysis report
```

## Methodology

### Seven-Layer Cross-Validation Framework

Each factor undergoes validation across 7 independent layers before acceptance:

1. **t-test** — Welch's t-test for return differentials
2. **Mann-Whitney U** — Non-parametric robustness check
3. **Logit regression** — Multivariate factor validation
4. **Quintile monotonicity** — Factor return monotonicity across 5 bins
5. **Industry × factor matrix** — Cross-industry consistency
6. **Market-cap stratification** — Directional stability across cap layers
7. **Cross-year stability** — 6-year rolling validation (direction > magnitude)

### Design Principles

- **Forward-looking bias prevention**: All signals use `prior_year = year - 1`, strict temporal alignment
- **Survivorship bias awareness**: Early-year survival rates as low as 32.2% — findings framed as "stability in surviving samples" rather than absolute claims
- **Cache integrity checks**: `safe_read_cache()` validates cached data before use, auto-purges corrupted files
- **Uncertainty labeling**: Every conclusion explicitly states data limitations, parameter sensitivity, and untested assumptions

## Key Findings

### Factor direction flips by market cap

| Factor | Micro-cap (<3B) | Mid-cap | Large-cap (>100B) |
|--------|-----------------|---------|-------------------|
| Low volatility | Positive (anti-fragile) | Neutral | Negative |
| Low PB | Negative (not safety) | Positive | Positive |
| High ROE | Negative | Positive | Strong positive |

**Implication**: Factor research must account for the structural differences of the underlying stock pool. Applying textbook large-cap logic to micro-caps leads to systematic losses.

### Directional consistency > Statistical significance

- Momentum factor: avg |t| = 3.06, but 3 positive + 3 negative years over 6 years → **unreliable**
- Low volatility factor: avg |t| = 3.28, but 6/6 years consistent direction → **reliable**

### Q7: ML-based profitable micro-cap discovery

**Method**: 401 stocks × 59 features (20 price + 7 fundamental + 4 derived + 28 industry). Three parallel approaches: UMAP + HDBSCAN clustering, XGBoost + SHAP, Autoencoder latent space. Cross-validated with feature ablation to isolate circular reasoning from genuine signal.

**Key results** (corrected for overfitting):

| Model | Train AUC | Test AUC | Gap |
|-------|-----------|----------|-----|
| XGBoost (original) | 1.000 | 0.977 | 0.023 |
| XGBoost (fixed, strong regularization) | 0.992 | 0.977 | 0.015 |
| XGBoost (no circular features) | **0.984** | **0.982** | **0.002** |
| Logistic (baseline) | 0.970 | 0.961 | 0.009 |

- Pure price signal AUC: 0.886 — market already prices in much of the quality difference
- Circular leakage premium: ~0.003 (full features − no-circular) — smaller than initially feared
- Feature importance: `pcv` (profit variability), `nm` (net margin), `rev_growth` dominate

**Limitations acknowledged**:
- 401 samples is small for 59-dim feature space
- Tier labels derived from net profit — `eps`/`roe`/`np_growth` carry conceptual leakage
- Train/test gap of ~0.01 remains even after regularization
- No out-of-time validation beyond the available window

### Q8: Market-cap factor attribution

Factor effectiveness systematically varies by market-cap regime. Volatility is negative for micro-caps (low-vol stocks more resilient) but positive for large-caps (volatility = growth opportunity). This suggests no universal factor definition works across all market-cap layers.

### Q9: Interaction effects

Single-dimension analysis misses critical interactions. Example: low-vol × high turnover micro-caps show 61.3% resilience, vs. low-vol × low turnover at 28.6%. The interaction effect rivals the main effect in magnitude.

## "Alpha is Constraint Rent" — The Framework

[alpha_meaning.md](alpha_meaning.md) develops an original framework for understanding alpha:

**Alpha is not prediction accuracy.** Alpha is the rent collected from market participants who are forced to trade at disadvantageous prices due to institutional, regulatory, liquidity, or behavioral constraints.

Six types of alpha rents:
1. **Information rent** — knowing what others don't (hardest to sustain)
2. **Liquidity rent** — providing immediacy when others must trade
3. **Inventory rent** — bearing short-term risk others cannot
4. **Institutional constraint rent** — profiting from rules others must follow
5. **Time-option rent** — charging for the right to trade now vs. later
6. **Behavioral rent** — predictable human responses to stress, drawdown, and career risk

**Critical insight**: Mistakes get corrected. Constraints recur. Sustainable alpha is built on constraints, not errors.

## Post-Mortem: Lessons Learned

[项目复盘_错误与成长.md](项目复盘_错误与成长.md) documents methodology failures encountered during the project:

- **AI-assisted research boundary**: AI-generated conclusions must be cross-checked against raw outputs — text-to-data alignment is a manual step
- **Parameter thresholds**: Using AI-suggested cutoffs (e.g., fixed asset ratio > 30%) without verifying against data distribution
- **Forward-looking bias**: Using current-year daily data to predict current-year returns
- **Cross-module consistency**: Independent module conclusions contradicted each other — global consistency must be verified manually
- **Encoding failures**: Chinese character replacement silently failing due to CJK encoding mismatches

## Setup

```bash
pip install numpy pandas scipy matplotlib xgboost scikit-learn umap-learn hdbscan torch akshare baostock statsmodels
```

Notebooks expect a `research_cache/` directory at the project root. Data is fetched via akshare (East Money) and baostock APIs.

## Limitations

- All analysis is in-sample relative to the available data window (through May 2026)
- No live trading or out-of-sample paper trading validation
- Micro-cap universe (861520.EI) has extreme survivorship bias — early-year samples are systematically stronger
- Factor findings are specific to A-share market structure; generalizability to other markets is untested
- This is research, not investment advice — historical patterns ≠ future returns

## Author

Quantitative finance intern project, 2026. Background: Mathematics (undergraduate + graduate), self-taught programming, AI-augmented research workflow.

---

*"The most important question in quant research is not 'does this signal work?' but 'who is paying, why can't they stop, and what risk am I taking?'"*
