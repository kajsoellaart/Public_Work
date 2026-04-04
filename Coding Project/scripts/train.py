"""
train.py

Train a LightGBM LambdaMART ranking model on engineered hotel search features.

The relevance target is graded: booking_bool=1 → 5, click_bool=1 → 1, else 0.
Hyperparameters are tuned via Bayesian search (Hyperopt) on a 40K-query sample,
then the final model is trained on the full dataset and saved with a manifest.

Usage:
    python train.py --parquet <path> --features <json> [--n-splits 5] [--n-trials 500] [--seed 42]
    python train.py --parquet data/features/training_features_snappy.parquet --features feature_config.json --skip-tune

Outputs (timestamped):
    lambdamart_<ts>.txt              — Booster in LightGBM text format
    lambdamart_<ts>.pkl              — Serialized LGBMRanker (joblib)
    lambdamart_<ts>_manifest.json    — Feature list, hyperparameters, best_iteration
    best_hyperparams.json            — Best hyperparameters from tuning run
"""

import argparse
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "relevance"
GROUP_COL  = "srch_id"

EARLY_STOP_ROUNDS = 100


def load_features(feature_json: str) -> list:
    """Load and deduplicate the selected feature list from JSON."""
    with open(feature_json, "r") as f:
        config = json.load(f)
    raw_feats = config.get("features", [])
    seen, features = set(), []
    for feat in raw_feats:
        if feat not in seen:
            seen.add(feat)
            features.append(feat)
    logger.info(f"Loaded {len(features)} features from {feature_json}.")
    return features


def load_data(parquet_path: str, features: list) -> tuple:
    """Stream-load parquet in row-group chunks, returning (X, y, groups)."""
    load_cols = list(dict.fromkeys(features + [TARGET_COL, GROUP_COL]))
    pf = pq.ParquetFile(parquet_path)

    missing = [f for f in features if f not in pf.schema_arrow.names]
    if missing:
        raise RuntimeError(f"Features not found in Parquet: {missing}")

    dfs = []
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=load_cols)
        dfs.append(tbl.to_pandas())
        logger.info(f"Row group {rg + 1}/{pf.num_row_groups} loaded.")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Data shape: {df.shape}")
    return df[features], df[TARGET_COL], df[GROUP_COL]


def optimize_hyperparams(X, y, groups, base_params: dict, n_trials: int, n_splits: int, seed: int) -> dict:
    """
    Bayesian hyperparameter search (Hyperopt TPE) on a 40K-query subsample.
    Returns the best hyperparameter dict to merge into base_params.
    """
    param_space = {
        "learning_rate":     hp.loguniform("learning_rate",     np.log(0.005), np.log(0.07)),
        "num_leaves":        hp.choice("num_leaves",            [31, 63, 127]),
        "max_depth":         hp.choice("max_depth",             [4, 6, 8, 10, 12, -1]),
        "min_data_in_leaf":  hp.quniform("min_data_in_leaf",    10, 100, 5),
        "feature_fraction":  hp.uniform("feature_fraction",     0.4, 1.0),
        "bagging_fraction":  hp.uniform("bagging_fraction",     0.4, 1.0),
        "bagging_freq":      hp.choice("bagging_freq",          [0, 5, 10]),
        "lambda_l1":         hp.uniform("lambda_l1",            0.0, 5.0),
        "lambda_l2":         hp.uniform("lambda_l2",            0.0, 5.0),
        "min_gain_to_split": hp.uniform("min_gain_to_split",    0.0, 1.0),
    }

    # Sample up to 40K queries for speed
    rng = np.random.RandomState(seed)
    sample_q = rng.choice(groups.unique(), size=min(40000, groups.nunique()), replace=False)
    mask = groups.isin(sample_q)
    X_sub, y_sub, g_sub = X[mask], y[mask], groups[mask]
    logger.info(f"Tuning on {g_sub.nunique()} queries ({len(X_sub)} rows).")

    trials = Trials()

    def objective(hparams):
        hparams["num_leaves"]       = int(hparams["num_leaves"])
        hparams["max_depth"]        = int(hparams["max_depth"])
        hparams["min_data_in_leaf"] = int(hparams["min_data_in_leaf"])
        hparams["bagging_freq"]     = int(hparams["bagging_freq"])

        params = {**base_params, **hparams}
        cv = GroupKFold(n_splits=n_splits)
        fold_scores = []

        for tr_idx, va_idx in cv.split(X_sub, y_sub, g_sub):
            X_tr, X_va = X_sub.iloc[tr_idx], X_sub.iloc[va_idx]
            y_tr, y_va = y_sub.iloc[tr_idx], y_sub.iloc[va_idx]
            g_tr = g_sub.iloc[tr_idx].value_counts().sort_index().values
            g_va = g_sub.iloc[va_idx].value_counts().sort_index().values

            model = lgb.LGBMRanker(**params)
            model.fit(
                X_tr, y_tr,
                group=g_tr,
                eval_set=[(X_va, y_va)],
                eval_group=[g_va],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS),
                    lgb.log_evaluation(period=100),
                ],
            )

            preds = model.predict(X_va, num_iteration=model.best_iteration_)
            qidx = g_sub.iloc[va_idx].groupby(g_sub.iloc[va_idx]).indices
            ndcgs = [
                ndcg_score(
                    y_va.iloc[idcs].values.reshape(1, -1),
                    preds[idcs].reshape(1, -1),
                    k=5,
                )
                for idcs in qidx.values()
                if len(idcs) > 1
            ]
            fold_scores.append(np.mean(ndcgs))

        return {"loss": -np.mean(fold_scores), "status": STATUS_OK}

    best_hps = fmin(fn=objective, space=param_space, algo=tpe.suggest,
                    max_evals=n_trials, trials=trials)
    best_params = space_eval(param_space, best_hps)

    for key in ("num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq"):
        best_params[key] = int(best_params[key])

    logger.info(f"Best hyperparameters: {best_params}")
    return best_params


def cross_validate(X, y, groups, params: dict, n_splits: int) -> list:
    """GroupKFold cross-validation reporting NDCG@5 per fold."""
    kf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        g_train = groups.iloc[train_idx].value_counts().sort_index().values
        g_val   = groups.iloc[val_idx].value_counts().sort_index().values

        ranker = lgb.LGBMRanker(**params)
        ranker.fit(
            X_train, y_train,
            group=g_train,
            eval_set=[(X_val, y_val)],
            eval_group=[g_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS),
                lgb.log_evaluation(period=100),
            ],
        )

        preds = ranker.predict(X_val, num_iteration=ranker.best_iteration_)
        ndcgs = []
        for _, idx in groups.iloc[val_idx].groupby(groups.iloc[val_idx]).indices.items():
            rel_true = y_val.iloc[idx].values.reshape(1, -1)
            rel_pred = preds[idx].reshape(1, -1)
            if rel_true.shape[1] > 1:
                ndcgs.append(ndcg_score(rel_true, rel_pred, k=5))

        fold_ndcg = np.mean(ndcgs)
        fold_metrics.append(fold_ndcg)
        logger.info(f"Fold {fold} NDCG@5: {fold_ndcg:.5f}")

    logger.info(f"CV NDCG@5: {np.mean(fold_metrics):.5f} ± {np.std(fold_metrics):.5f}")
    return fold_metrics


def train_final(X, y, groups, params: dict, features: list) -> None:
    """Train on the full dataset and save model + manifest (timestamped)."""
    logger.info("Training final model on full dataset...")
    final_ranker = lgb.LGBMRanker(**params)
    full_groups  = groups.value_counts().sort_index().values

    final_ranker.fit(X, y, group=full_groups, callbacks=[lgb.log_evaluation(100)])

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lambdamart_{ts}"

    final_ranker.booster_.save_model(f"{model_name}.txt",
                                     num_iteration=final_ranker.best_iteration_)
    joblib.dump(final_ranker, f"{model_name}.pkl")

    manifest = {
        "timestamp":      ts,
        "features":       features,
        "params":         params,
        "best_iteration": final_ranker.best_iteration_,
    }
    with open(f"{model_name}_manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    logger.info(f"Saved {model_name}.txt / .pkl / _manifest.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LambdaMART hotel search ranker")
    parser.add_argument("--parquet",    required=True, help="Path to engineered feature Parquet file")
    parser.add_argument("--features",   required=True, help="Path to JSON with selected feature list")
    parser.add_argument("--n-splits",   type=int, default=5,   help="GroupKFold splits (default: 5)")
    parser.add_argument("--n-trials",   type=int, default=500, help="Hyperopt trials (default: 500)")
    parser.add_argument("--seed",       type=int, default=42,  help="Random seed (default: 42)")
    parser.add_argument("--skip-tune",  action="store_true",   help="Skip hyperparameter search")
    return parser.parse_args()


def main():
    args = parse_args()

    features     = load_features(args.features)
    X, y, groups = load_data(args.parquet, features)

    base_params = {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [5],
        "boosting_type":    "gbdt",
        "learning_rate":    0.1,
        "num_leaves":       31,
        "max_depth":        12,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.6,
        "bagging_freq":     5,
        "lambda_l1":        1.0,
        "lambda_l2":        1.0,
        "n_estimators":     3000,
        "random_state":     args.seed,
        "verbose":          -1,
    }

    if not args.skip_tune:
        best_params = optimize_hyperparams(
            X, y, groups, base_params,
            n_trials=args.n_trials,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        base_params.update(best_params)
        with open("best_hyperparams.json", "w") as f:
            json.dump(best_params, f, indent=2)

    cross_validate(X, y, groups, base_params, n_splits=args.n_splits)
    train_final(X, y, groups, base_params, features)


if __name__ == "__main__":
    main()
