import warnings
import os

import pandas as pd

from scripts.schema import DatasetSchema


RAW_TRAINING_CSV       = "data/raw/training_set_VU_DM.csv"
PROPERTY_AGGREGATES    = "data/features/property_aggregates.parquet"
TRAINING_FEATURES      = "data/features/training_features.parquet"
TRAINING_FEATURES_SNAPPY = "data/features/training_features_snappy.parquet"


class PropertyAggregation:

    @staticmethod
    def run():
        """Run the full feature engineering pipeline, skipping steps whose outputs already exist."""
        if not os.path.exists(PROPERTY_AGGREGATES):
            PropertyAggregation.build_property_features()
        if not os.path.exists(TRAINING_FEATURES):
            PropertyAggregation.merge_into_training_set()
        if not os.path.exists(TRAINING_FEATURES_SNAPPY):
            PropertyAggregation.compress()

    @staticmethod
    def build_property_features():
        """
        Aggregate raw training data at the property level.

        For each numerical and ordinal column, computes mean, median, std, min, max,
        sum, Q25, and Q75 across all search impressions for that property.
        Binary columns are aggregated by mean (click/booking rates).
        ID columns are aggregated by mode.
        Result is saved as property_aggregates.parquet.
        """
        print("Building property-level aggregated features...")
        warnings.filterwarnings('ignore')

        df = pd.read_csv(RAW_TRAINING_CSV, parse_dates=['date_time'])
        df['search_year']      = df['date_time'].dt.year
        df['search_month']     = df['date_time'].dt.month
        df['search_dayofweek'] = df['date_time'].dt.dayofweek

        num_ord_columns = DatasetSchema.numerical_columns + DatasetSchema.ordinal_columns

        agg_funcs = [
            ('mean',     'mean'),
            ('median',   'median'),
            ('std',      'std'),
            ('min',      'min'),
            ('max',      'max'),
            ('sum',      'sum'),
            (lambda x: x.quantile(0.25), 'quantile25'),
            (lambda x: x.quantile(0.75), 'quantile75'),
        ]

        agg_list = {}
        for col in num_ord_columns:
            for func, name in agg_funcs:
                agg_list[f"{col}_{name}"] = (col, func)

        property_agg = df.groupby('prop_id').agg(**{
            new_col: pd.NamedAgg(column=col, aggfunc=func)
            for new_col, (col, func) in agg_list.items()
        })

        binary_agg = df.groupby('prop_id')[DatasetSchema.binary_nominal_columns].mean()
        binary_agg.columns = [f"{col}_mean" for col in binary_agg.columns]

        # Mode-aggregate ID columns (excluding srch_id which varies per row)
        id_columns = [col for col in DatasetSchema.id_columns if col != 'srch_id']
        id_mode = df.groupby('prop_id')[id_columns].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA
        )

        property_df = pd.concat([property_agg, binary_agg, id_mode], axis=1)
        os.makedirs(os.path.dirname(PROPERTY_AGGREGATES), exist_ok=True)
        property_df.to_parquet(PROPERTY_AGGREGATES)
        print(f"Property features saved ({property_df.shape[1]} columns).")

    @staticmethod
    def merge_into_training_set():
        """
        Merge property-level aggregated features into the training set.
        All property feature columns are prefixed with 'propf_' to distinguish
        them from the row-level original features.
        """
        print("Merging property features into training set...")
        warnings.filterwarnings('ignore')

        df = pd.read_csv(RAW_TRAINING_CSV, parse_dates=['date_time'])
        df['search_year']      = df['date_time'].dt.year
        df['search_month']     = df['date_time'].dt.month
        df['search_dayofweek'] = df['date_time'].dt.dayofweek

        property_df = pd.read_parquet(PROPERTY_AGGREGATES)
        property_df_prefixed = property_df.add_prefix("propf_")

        df = df.merge(property_df_prefixed, how='left', left_on='prop_id', right_on='propf_prop_id')
        df.drop(columns=["propf_prop_id"], inplace=True)

        df.to_parquet(TRAINING_FEATURES)
        print(f"Merged dataset saved ({df.shape[1]} columns, {len(df)} rows).")

    @staticmethod
    def compress():
        """Recompress the merged dataset with Snappy for faster downstream I/O."""
        print("Compressing to Snappy parquet...")
        df = pd.read_parquet(TRAINING_FEATURES)
        df.to_parquet(TRAINING_FEATURES_SNAPPY, compression="snappy")
        print("Done.")
