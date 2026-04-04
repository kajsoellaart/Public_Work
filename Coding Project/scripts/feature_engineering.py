import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the pipeline.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def load_data(input_path: Path, parse_dates: list = ["date_time"]) -> pd.DataFrame:
    """
    Load raw CSV data into a DataFrame.

    Args:
        input_path: Path to the raw CSV file.
        parse_dates: List of columns to parse as datetime.

    Returns:
        pd.DataFrame: Loaded data.
    """
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, parse_dates=parse_dates)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create or transform features for the pipeline.

    Args:
        df: Original DataFrame.

    Returns:
        DataFrame with new features.
    """
    logging.info("Starting feature engineering")
    df = df.copy()

    # Temporal features
    df['month'] = df['date_time'].dt.month
    df['week'] = df['date_time'].dt.isocalendar().week.astype(int)

    # Price features
    df['price_cleaned'] = df['price_usd']
    df['length_ok'] = df['srch_length_of_stay'] > 0
    df['room_ok'] = df['srch_room_count'] > 0
    df['total_occupants'] = df['srch_adults_count'] + df['srch_children_count']
    df['occupants_ok'] = df['total_occupants'] > 0

    df['price_rank'] = (
        df.groupby('srch_id')['price_cleaned']
          .rank(method='dense', ascending=True)
          .astype(int)
    )
    max_rank = df.groupby('srch_id')['price_rank'].transform('max')
    df['price_pct_rank'] = np.where(
        max_rank > 1,
        (df['price_rank'] - 1) / (max_rank - 1),
        0.0
    )

    df['price_per_night'] = np.where(
        df['length_ok'],
        df['price_cleaned'] / df['srch_length_of_stay'],
        np.nan
    )
    df['price_per_room'] = np.where(
        df['room_ok'],
        df['price_cleaned'] / df['srch_room_count'],
        np.nan
    )
    df['price_per_person'] = np.where(
        df['occupants_ok'],
        df['price_cleaned'] / df['total_occupants'],
        np.nan
    )
    df['price_per_person_per_night'] = np.where(
        df['length_ok'] & df['occupants_ok'],
        df['price_cleaned'] / (df['total_occupants'] * df['srch_length_of_stay']),
        np.nan
    )

    df['star_zero_flag'] = (df['prop_starrating'] == 0).astype(int)
    df['price_per_star'] = np.where(
        df['prop_starrating'] > 0,
        df['price_cleaned'] / df['prop_starrating'],
        np.nan
    )
    df['review_missing_flag'] = df['prop_review_score'].isna().astype(int)
    df['review_zero_flag'] = (df['prop_review_score'] == 0).astype(int)
    df['has_review'] = df['prop_review_score'] > 0
    df['price_per_review'] = np.where(
        df['has_review'],
        df['price_cleaned'] / df['prop_review_score'],
        np.nan
    )

    df['price_trend'] = (
        df.sort_values(['prop_id', 'date_time'])
          .groupby('prop_id')['price_cleaned']
          .pct_change()
          .sort_index()
    )

    # Rating features
    df['star_rank'] = (
        df.groupby('srch_id')['prop_starrating']
          .rank(method='dense', ascending=False)
          .astype(int)
    )
    df['review_score_for_rank'] = df['prop_review_score'].where(df['prop_review_score'] > 0, -1)
    df['review_rank'] = (
        df.groupby('srch_id')['review_score_for_rank']
          .rank(method='dense', ascending=False)
          .astype(int)
    )
    df['star_discrepancy'] = df['prop_starrating'] - df['visitor_hist_starrating']
    df['hist_star_missing_flag'] = df['visitor_hist_starrating'].isna().astype(int)
    df['hist_price_missing_flag'] = df['visitor_hist_adr_usd'].isna().astype(int)

    # Location features
    w1, w2 = 0.6, 0.4
    df['merged_prop_location_score'] = (
        df['prop_location_score1'] * w1 + df['prop_location_score2'] * w2
    )
    df['merged_prop_location_score'] = df['merged_prop_location_score'].where(
        df['prop_location_score1'].notna() & df['prop_location_score2'].notna(),
        df['prop_location_score1'].fillna(df['prop_location_score2'])
    )
    df['within_country'] = (
        (df['visitor_location_country_id'] == df['prop_country_id']).astype(int)
    )
    df['closest_hist_price'] = (
        np.abs(df['price_per_night'] - df['visitor_hist_adr_usd'])
    )

    # User group features
    df['full_family'] = np.where(
        (df['srch_adults_count'] >= 2) & (df['srch_children_count'] >= 1),
        1, 0
    )
    df['single_parent_family'] = np.where(
        (df['srch_adults_count'] == 1) & (df['srch_children_count'] >= 1),
        1, 0
    )

    # Competitor-comparison features
    rate_cols = [f'comp{i}_rate' for i in range(1, 9)]
    inv_cols = [f'comp{i}_inv' for i in range(1, 9)]
    diff_cols = [f'comp{i}_rate_percent_diff' for i in range(1, 9)]

    # Count reporting competitors
    comp_presence = [
        df[rate].notna() | df[inv].notna() | df[diff].notna()
        for rate, inv, diff in zip(rate_cols, inv_cols, diff_cols)
    ]
    df['comp_data_count'] = np.sum(comp_presence, axis=0)

    # Raw counts
    df['comp_lower_rate'] = df[rate_cols].eq(1).sum(axis=1)
    df['comp_higher_rate'] = df[rate_cols].eq(-1).sum(axis=1)
    df['comp_availability'] = df[inv_cols].eq(0).sum(axis=1)

    # Ratios of counts to reporting competitors
    df['comp_lower_rate_ratio'] = np.where(
        df['comp_data_count'] > 0,
        df['comp_lower_rate'] / df['comp_data_count'],
        0.0
    )
    df['comp_higher_rate_ratio'] = np.where(
        df['comp_data_count'] > 0,
        df['comp_higher_rate'] / df['comp_data_count'],
        0.0
    )
    df['comp_availability_ratio'] = np.where(
        df['comp_data_count'] > 0,
        df['comp_availability'] / df['comp_data_count'],
        0.0
    )

    # Aggregated percent-diff stats
    df['comp_rate_percent_diff'] = df[diff_cols].mean(axis=1, skipna=True)
    df['comp_rate_percent_std'] = df[diff_cols].std(axis=1, ddof=0, skipna=True)

    df['no_competitors'] = (df['comp_data_count'] == 0).astype(int)

    # Documents count
    df['n_documents'] = df.groupby('srch_id')['prop_id'].transform('count')

    # Cleanup intermediate columns
    drop_cols = [
        'length_ok', 'room_ok', 'occupants_ok', 'total_occupants', 'has_review',
        'review_score_for_rank', 'comp_data_count'
    ]
    existing = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing, inplace=True)

    logging.info("Feature engineering completed")
    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the processed DataFrame.

    Args:
        df: DataFrame to save.
        output_path: Path to write the CSV.
    """
    logging.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature engineering for VU DM dataset"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw input CSV file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save engineered CSV file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    df_raw = load_data(args.input)
    df_features = engineer_features(df_raw)
    save_data(df_features, args.output)


if __name__ == "__main__":
    main()
