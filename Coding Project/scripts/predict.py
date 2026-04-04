"""
predict.py

Load a trained LightGBM LambdaMART model and its manifest, score the test set,
and produce a Kaggle-style ranking submission CSV (columns: srch_id, prop_id).

Usage:
    python predict.py --model <model.pkl> --manifest <manifest.json> --test <test.parquet>
    python predict.py --model lambdamart_20250517.pkl --manifest lambdamart_20250517_manifest.json \
                      --test data/raw/test_set_VU_DM.parquet --output submission.csv
"""

import argparse
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GROUP_COL = "srch_id"
ID_COL    = "prop_id"
SCORE_COL = "score"


def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if "features" not in manifest:
        raise KeyError(f"'features' key not found in {manifest_path}")
    return manifest


def load_test_data(parquet_path: str, features: list) -> pd.DataFrame:
    """Stream-load test parquet in row-group chunks, filling missing features with NaN."""
    load_cols = list(dict.fromkeys(features + [GROUP_COL, ID_COL]))
    pf = pq.ParquetFile(parquet_path)

    chunks = []
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i, columns=load_cols)
        chunks.append(tbl.to_pandas())
        logger.info(f"Row group {i + 1}/{pf.num_row_groups} loaded.")

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Test DataFrame shape: {df.shape}")

    missing = [c for c in features if c not in df.columns]
    if missing:
        logger.warning(f"Features missing from test data (filling with NaN): {missing}")
        for c in missing:
            df[c] = np.nan

    return df


def predict_and_rank(model, df: pd.DataFrame, features: list, best_iteration: int = None) -> pd.DataFrame:
    """Score each row and rank properties within each search session."""
    X_test = df[features]
    if best_iteration is not None:
        preds = model.predict(X_test, num_iteration=best_iteration)
    else:
        preds = model.predict(X_test)

    df = df.copy()
    df[SCORE_COL] = preds
    df.sort_values([GROUP_COL, SCORE_COL], ascending=[True, False], inplace=True)
    return df[[GROUP_COL, ID_COL]].copy()


def parse_args():
    parser = argparse.ArgumentParser(description="Score test set with trained LambdaMART ranker")
    parser.add_argument("--model",    required=True, help="Path to model pickle (.pkl)")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--test",     required=True, help="Path to test Parquet file")
    parser.add_argument(
        "--output",
        default=f"submission_{datetime.now():%Y%m%d_%H%M%S}.csv",
        help="Output CSV path (default: submission_<timestamp>.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    manifest     = load_manifest(args.manifest)
    features     = manifest["features"]
    best_iter    = manifest.get("best_iteration", None)
    logger.info(f"Loaded manifest: {len(features)} features, best_iteration={best_iter}")

    logger.info(f"Loading model from {args.model} ...")
    model = joblib.load(args.model)

    test_df    = load_test_data(args.test, features)
    submission = predict_and_rank(model, test_df, features, best_iter)

    submission.to_csv(args.output, index=False)
    logger.info(f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
