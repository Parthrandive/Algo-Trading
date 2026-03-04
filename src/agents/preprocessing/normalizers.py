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
        
        # Threshold parameters
        regime_threshold = self.parameters.get("regime_threshold", 2.0)
        anomaly_threshold = self.parameters.get("anomaly_threshold", 3.0)
        
        rolling = df[target_col].rolling(window=window, min_periods=1)
        mean = rolling.mean()
        std = rolling.std()
        
        # Handle zero division for stationary constants over period
        df[output_col] = (df[target_col] - mean) / std.replace(0, np.nan)
        df[output_col] = df[output_col].fillna(0.0)
        
        # Add threshold flags (1 if threshold met, else 0)
        df[f"{output_col}_is_regime_shift"] = (df[output_col].abs() >= regime_threshold).astype(int)
        df[f"{output_col}_is_anomaly"] = (df[output_col].abs() >= anomaly_threshold).astype(int)
        
        return df

class MinMaxNormalizer(TransformNode):
    @classmethod
    def get_expected_version(cls) -> str:
        return "1.0"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.parameters.get("window", 30)
        target_col = self.parameters.get("target_column", "value")
        output_col = self.parameters.get("output_column", f"{target_col}_minmax")
        
        rolling = df[target_col].rolling(window=window, min_periods=1)
        rolling_min = rolling.min()
        rolling_max = rolling.max()
        
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

        # Some snapshots legitimately omit a configured macro column; emit neutral flags.
        if target_col not in df.columns:
            df[output_col] = 0
            return df
        
        pct_change = df[target_col].pct_change(fill_method=None).fillna(0.0)
        
        conditions = [
            pct_change >= threshold,
            pct_change <= -threshold
        ]
        choices = [1, -1]
        
        # Numpy select replaces complex row application
        df[output_col] = np.select(conditions, choices, default=0)
        return df
