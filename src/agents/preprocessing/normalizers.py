import numpy as np
import pandas as pd

from src.agents.preprocessing.transform_graph import TransformNode

class ZScoreNormalizer(TransformNode):
    @classmethod
    def get_expected_version(cls) -> str:
        return "1.0"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.parameters.get("window", 30)
        target_col = self.parameters.get("target_column", "value")
        output_col = self.parameters.get("output_column", f"{target_col}_zscore")
        
        mean = df[target_col].rolling(window=window, min_periods=1).mean()
        std = df[target_col].rolling(window=window, min_periods=1).std()
        
        # Handle zero division for stationary constants over period
        df[output_col] = (df[target_col] - mean) / std.replace(0, np.nan)
        df[output_col] = df[output_col].fillna(0.0)
        return df

class MinMaxNormalizer(TransformNode):
    @classmethod
    def get_expected_version(cls) -> str:
        return "1.0"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.parameters.get("window", 30)
        target_col = self.parameters.get("target_column", "value")
        output_col = self.parameters.get("output_column", f"{target_col}_minmax")
        
        rolling_min = df[target_col].rolling(window=window, min_periods=1).min()
        rolling_max = df[target_col].rolling(window=window, min_periods=1).max()
        
        denominator = rolling_max - rolling_min
        df[output_col] = (df[target_col] - rolling_min) / denominator.replace(0, np.nan)
        df[output_col] = df[output_col].fillna(0.0)
        return df

class LogReturnNormalizer(TransformNode):
    @classmethod
    def get_expected_version(cls) -> str:
        return "1.0"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        target_col = self.parameters.get("target_column", "close")
        output_col = self.parameters.get("output_column", f"{target_col}_log_return")
        
        # Avoid log(0) or negative price errors structurally
        safe_series = df[target_col].mask(df[target_col] <= 0)
        df[output_col] = np.log(safe_series / safe_series.shift(1))
        df[output_col] = df[output_col].fillna(0.0)
        return df

class DirectionalChangeDetector(TransformNode):
    @classmethod
    def get_expected_version(cls) -> str:
        return "1.0"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        target_col = self.parameters.get("target_column", "value")
        output_col = self.parameters.get("output_column", f"{target_col}_directional_flag")
        threshold = self.parameters.get("threshold", 0.05) # 5% change
        
        pct_change = df[target_col].pct_change().fillna(0.0)
        
        conditions = [
            pct_change >= threshold,
            pct_change <= -threshold
        ]
        choices = [1, -1]
        
        # Numpy select replaces complex row application
        df[output_col] = np.select(conditions, choices, default=0)
        return df
