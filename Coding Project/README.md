# Hotel Search Ranking — VU Amsterdam Data Mining 2025

**Result: top 5 out of ~400 students** in the VU Amsterdam Data Mining Techniques competition (2025).

## Task

Given a dataset of hotel search sessions from Expedia, predict the ranking of properties within each session to maximise NDCG@5. Each row represents one (search, property) impression; the targets are `click_bool` and `booking_bool`.

## Approach

The model is a **LightGBM LambdaMART** ranker with a graded relevance target (booking = 5, click = 1, else 0). Feature engineering runs in two stages: first, each property's historical statistics are aggregated across all its impressions (mean, median, std, min, max, Q25, Q75 for every numerical and ordinal column, prefixed `propf_`); second, per-row derived features are computed covering price normalisation, within-session rank, star/review metrics, competitor pricing comparisons, and user-group flags (~140 features total). Hyperparameters were tuned via Bayesian search (Hyperopt TPE, 500 trials on a 40K-query subsample) and evaluated with 5-fold group cross-validation stratified by `srch_id`.

## Repository structure

```
schema.py                  — dataset column schema (type lists used across modules)
feature_engineering.py     — per-row derived features (price, ratings, competitors, user groups)
property_aggregation.py    — property-level feature aggregation pipeline (propf_* features)
eda.py                     — column statistics on the engineered feature set
pipeline.py                — runs feature engineering + EDA end-to-end
train.py                   — hyperparameter search, cross-validation, final model training
predict.py                 — score test set and write submission CSV
feature_config.json        — selected feature list used for the final model submission
```

## Data

The dataset is the Expedia LETOR dataset provided by VU Amsterdam and is not included in this repository. Place the files at:

```
data/raw/training_set_VU_DM.csv
data/raw/test_set_VU_DM.csv
```

## Note on runnability

**This repository is intentionally not in a runnable state.** The raw data files, intermediate feature parquets, and trained model artifacts are excluded from version control. This is deliberate — the assignment is still active at VU Amsterdam and making a fully self-contained, copy-paste-runnable solution publicly available could enable academic dishonesty.

If you are a future student at VU Amsterdam: please do the work yourself. The approach section above describes the high-level method; the code is here for transparency, not to be submitted as your own.