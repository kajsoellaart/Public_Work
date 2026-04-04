import os

import pandas as pd
import numpy as np


TRAINING_FEATURES_SNAPPY = "data/features/training_features_snappy.parquet"
EDA_OUTPUT_CSV           = "data/eda/feature_statistics.csv"


class EDA:

    @staticmethod
    def run():
        """Load the compressed feature parquet and compute basic column statistics."""
        df = pd.read_parquet(TRAINING_FEATURES_SNAPPY, engine='pyarrow')
        EDA.column_statistics(df)

    @staticmethod
    def column_statistics(df):
        """Write per-column summary statistics (count, mean, std, min, max, NaN count) to CSV."""
        os.makedirs(os.path.dirname(EDA_OUTPUT_CSV), exist_ok=True)
        numeric_df = df.select_dtypes(include=[np.number])

        statistics = []
        for col in numeric_df.columns:
            values = numeric_df[col]
            statistics.append({
                'Variable':  col,
                'Count':     len(values),
                'Mean':      values.mean(),
                'Std':       values.std(),
                'Min':       values.min(),
                'Max':       values.max(),
                'NaN Count': values.isna().sum(),
            })

        pd.DataFrame(statistics).to_csv(EDA_OUTPUT_CSV, index=False)
