# E-CUP 2026 — Task 3: Search User LTV/GMV

Predict an Ozon user's total order GMV (Gross Merchandise Value) over the next 30 days from daily activity in Search and Catalog. Competition: E-CUP 2026 on ods.ai.

**Public LB: 14th place** (1.6466129691) · **Private LB: 27th place** (1.6637240734)

- **Metric**: RMSLE (`log1p`, negative predictions clipped to zero).
- **Data**: ~30.6M anonymized daily activity records for 250,000 users, `2025-01-01 → 2026-02-13`.
- **Target**: each user's total `gmv` for `2026-02-14 → 2026-03-15`.
- **Submission**: CSV with `user_id,predict` columns for all 250,000 users.

The absolute public-to-private score shift (+0.0171) is not meaningful on its own: the private set was harder, and all participants' scores worsened. The meaningful change is the relative drop from 14th to 27th, a loss of 13 places.

## Repository structure

```
configs/  data, feature, CV, and trained model variant parameters (YAML)
src/      pipeline code (ecup_ltv): data loading, features, training, inference, blending
scripts/  entry points: build_features.py, train.py, predict.py, blend.py, train_catboost.py, multi_fold_eval.py
```

This is the core solution: a reproducible LightGBM/CatBoost pipeline using anchor-based features and blending. One-off research scripts (a panel CNN, a CNN + tabular feature hybrid, and ~90 leaderboard probes of individual hypotheses and directions) are outside this repository. They contributed to the final result but are not required to reproduce the working pipeline. See “What worked” below for their measurable contributions.

## Reproduction

### 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data

Place the competition files in `data/raw/`:

```
data/raw/train.parquet          # raw activity log (the Data page on ods.ai)
data/raw/sample_submit.csv      # user_id list for inference
```

The data is not included: we do not own the intellectual property rights to redistribute it. This repository contains only the code that processes it.

### 3. Features

```bash
python scripts/build_features.py
```

Anchor-based features (rolling daily windows) are built for all CV folds and the final anchor, then cached as Parquet files in `artifacts/features_store/`.

### 4. Training and inference

```bash
python scripts/train.py train_v3_top100_5seed
python scripts/predict.py train_v3_top100_5seed
```

`train.py` computes the walk-forward CV metric and fits the final model on all folds. `predict.py` writes `artifacts/submissions/*.csv` and checks the `user_id` set against `data/raw/sample_submit.csv`.

> **CV caveat:** a post-competition audit found overlapping training/validation target windows with `stride_days=14` and `horizon_days=30`. The local CV scores in the configs are legacy estimates, rather than valid purged CV results. See “What was missing” below.

### 5. Blending and alternative models

```bash
python scripts/blend.py <run_a> <run_b> ... --submit
python scripts/train_catboost.py train_catboost_diverse
python scripts/multi_fold_eval.py train_v3_top100_5seed
```

## Method

**Features** (`src/ecup_ltv/features.py`): for each anchor date, rolling windows of `7/14/30/60/90/180d/all` plus YoY-30d (the same calendar period a year earlier, a leakage-free seasonality signal), over 12 of the 16 raw signal columns × 4 aggregations (`sum/mean/std/max`). Additional features include recency/tenure, calendar features, normalization against the platform trend, derived ratio/conversion features, and an RFM score.

**BTYD** (`src/ecup_ltv/btyd.py`): BG/NBD + Gamma-Gamma, a parametric model of purchase frequency and customer dropout probability, independent of gradient boosting. These features are strong individually but redundant with the existing recency/frequency features.

**CV** (`src/ecup_ltv/features.py: generate_cv_anchor_dates`): walk-forward anchor dates spaced 14 days apart, leaving a 30-day future target horizon and at least 120 days of history. An ordinary random split introduces leakage because features at date t use all history up to t.

**Models**: LightGBM (primary) and CatBoost. A panel convolution over raw daily sequences and a CNN + tabular feature hybrid were also explored; their research code is not included (see above). Target = `log1p(clip(gmv, 0))`; predictions use `expm1` and are clipped to ≥0.

**Ensembling** (`src/ecup_ltv/blend.py`): averaging several independently trained bundles in log1p space. Weights can be set manually or fitted using NNLS stacking on out-of-fold predictions.

**Tested but excluded from the final pipeline**: MLP, Tweedie objective, and DART boosting. None delivered a consistent improvement over a standard LightGBM regressor.

**The hurdle model as a direction rather than a standalone model**: an activity classifier plus a regressor for active users was rejected as a *model* (RMSLE 1.99 versus 1.665 for the main pipeline), a fair assessment of its standalone quality. However, measured as a *direction* in log space, it yielded +0.0541 at 7.3σ, the largest coefficient in the project. In a blend, the direction enters through

```
RMSLE(a)² = R₀² − 2a⟨e,d⟩ + a²⟨d,d⟩,   d = z_model − z_base
```

The coefficient's sign is determined by the direction of the errors, rather than standalone model quality. Rejecting a model and rejecting its direction are separate decisions; the former does not imply the latter.

## Final results and retrospective

```
public   1.6466129691   14th place
private  1.6637240734   27th place
```

Exposure estimates recorded before submissions closed:

```
fitting penalty     ~20 coefficients x 2.09e-5   0.00042
selection inflation  best of ~100 submissions    0.00070
combined                                         0.00112
```

On the public leaderboard, places 9–16 were separated by 0.0006; losing 0.0011 costs roughly 13 positions. The risk estimate was already correct in order of magnitude before the deadline. The mistake was the decision: most effort went into fitting blend coefficients on the 50,000 public rows because this produced an immediate measurable result, whereas model development did not.

Of the total 0.0084 public improvement, roughly 0.0045 came from leaderboard algebra (a chain of more than twenty fitted blend coefficients), and only about 0.0039 from the models themselves (denser anchors, a panel CNN, and a feature hybrid). The model improvements transferred to the private set; the leaderboard fitting did not. Our own noise measurements had already indicated this.

The right strategy would have been to stop the joint optimization approach after the first two or three useful steps and spend the remaining time improving models rather than refitting blend weights.

**What was missing**: trustworthy local validation. Purged CV with an embargo was written only four days before the deadline and never made operational. Without it, every model hypothesis cost a leaderboard submission, with only five allowed per day. This encouraged blend algebra, which returned an answer in a minute without training.

**What worked and should transfer to a new task** (measured as a direction in log space; `a*` is the optimal coefficient):

| Direction | a* |
|---|---|
| Denser anchors (7-day rather than 14-day stride) | +0.0307 |
| Panel CNN over raw daily sequences | +0.0353 |
| Hybrid: neural network + tabular features | +0.0199 |
| Hurdle model as a direction | +0.0541 |

All four change the model itself rather than simply reweighting the blend.

## Measurement tools

The transferable part of this solution is the measurement methodology for a noisy leaderboard, rather than its features or hyperparameters:

**1. The leaderboard as algebra.** `RMSLE(a)² = R₀² − 2a⟨e,d⟩ + a²⟨d,d⟩` is an identity. A single probe at a known step `s` therefore determines the optimal coefficient exactly: `a* = ((R₀² − R_probe²) + s²)/(2s)`. On five consecutive occasions, the calculated and actual scores matched within 1e-6.

**2. Measure orthogonality against the span, rather than individual files.** For a new direction, the relevant quantity is the share of its variance outside the linear span of all previously measured directions: `res = d − M(MᵀM)⁻¹Mᵀd`. A direction correlated at 0.5 with each of four measured directions may still lie entirely within their span.

**3. Measure the training noise floor.** A run differing from the baseline only in its seeds gives `sd`, the level below which a direction is indistinguishable from randomness (0.04177 in this project). This separates real signal from seed noise without another leaderboard submission.

**4. The private-set penalty.** Each coefficient fitted on the public 50k out of 250k rows costs approximately 2.09e-5 on the private set (`sd(α_pub − α_priv) = 0.0083`). An honest upper bound on the number of fitted coefficients is the number of independent measurements, rather than the desired public score precision.

**5. Store the panel as one cube.** Panel models use daily counters without product IDs: 250,000 × 409 × 6 values, or 1.23 GB in float16, which fits entirely in memory. Each anchor is a slice of this prebuilt cube, avoiding a fresh allocation and densification of the log.

## Reproducibility

- All parameters live in `configs/*.yaml`; none are hardcoded in the code.
- All random seeds are fixed (`src/ecup_ltv/seed.py` and per-model seeds in the configs).
- Training (`train.py`) and inference (`predict.py`) are fully separated. Inference only reads a saved bundle.
- Each saved bundle includes its resolved config and feature list.
