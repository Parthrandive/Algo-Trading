import pandas as pd
import numpy as np

MACRO_COLUMNS = [
    "CPI",
    "WPI",
    "IIP",
    "FII_FLOW",
    "DII_FLOW",
    "FX_RESERVES",
    "INDIA_US_10Y_SPREAD",
    "RBI_BULLETIN",
    "REPO_RATE",
    "US_10Y",
]

def add_lag_features(df: pd.DataFrame, target_col: str, lags: int) -> pd.DataFrame:
    """
    Adds lagged features for a specific column.
    
    Args:
        df: Input DataFrame.
        target_col: The column to create lags for (e.g., 'close').
        lags: Number of lag periods to create.
        
    Returns:
        DataFrame with added lag columns.
    """
    result = df.copy()
    for lag in range(1, lags + 1):
        result[f'{target_col}_lag_{lag}'] = result[target_col].shift(lag)
    return result

def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    """
    Adds rolling mean (SMA) and rolling standard deviation features.
    
    Args:
        df: Input DataFrame.
        target_col: The column to compute rolling features on.
        windows: List of window sizes (e.g., [7, 14, 21]).
        
    Returns:
        DataFrame with added rolling features.
    """
    result = df.copy()
    for window in windows:
        result[f'{target_col}_roll_mean_{window}'] = result[target_col].rolling(window=window).mean()
        result[f'{target_col}_roll_std_{window}'] = result[target_col].rolling(window=window).std()
    return result

def add_rsi(df: pd.DataFrame, target_col: str = 'close', period: int = 14) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI).
    
    Args:
        df: Input DataFrame.
        target_col: The series to calculate RSI on.
        period: The RSI period (default 14).
    
    Returns:
        DataFrame with 'rsi' column.
    """
    result = df.copy()
    delta = result[target_col].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Prevent division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    
    rsi = 100 - (100 / (1 + rs))
    # If avg_loss was 0, RSI is 100
    rsi = rsi.fillna(100)
    
    result['rsi'] = rsi
    # Mask initial periods correctly
    result.loc[result.index[:period], 'rsi'] = np.nan
    return result

def add_macd(df: pd.DataFrame, target_col: str = 'close', fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD).
    
    Args:
        df: Input DataFrame.
        target_col: The series to calculate MACD on.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal EMA period.
    
    Returns:
        DataFrame with 'macd', 'macd_signal', and 'macd_hist' columns.
    """
    result = df.copy()
    
    ema_fast = result[target_col].ewm(span=fast, adjust=False).mean()
    ema_slow = result[target_col].ewm(span=slow, adjust=False).mean()
    
    result['macd'] = ema_fast - ema_slow
    result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
    result['macd_hist'] = result['macd'] - result['macd_signal']
    
    return result

def add_macro_regime_features(df: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    """
    Build a macro composite for technical models without fabricating history.

    - No backward fill before first real observation.
    - No forced zero for missing raw macro series.
    - Optional feature gating via `df.attrs["macro_excluded_features"]`.
    """
    result = df.copy()
    excluded_features = set(result.attrs.get("macro_excluded_features", []))
    macro_weights = {
        "CPI": -0.18,
        "WPI": -0.16,
        "IIP": 0.16,
        "FII_FLOW": 0.18,
        "DII_FLOW": 0.10,
        "FX_RESERVES": 0.12,
        "INDIA_US_10Y_SPREAD": -0.12,
        "REPO_RATE": -0.14,
        "US_10Y": -0.12,
        "RBI_BULLETIN": 0.04,
    }

    weighted_sum = pd.Series(0.0, index=result.index, dtype=float)
    available_weight = pd.Series(0.0, index=result.index, dtype=float)
    total_abs_weight = float(sum(abs(weight) for weight in macro_weights.values()))

    for col, weight in macro_weights.items():
        if col not in result.columns:
            result[col] = np.nan

        raw = pd.to_numeric(result[col], errors="coerce")
        if col in excluded_features:
            result[col] = np.nan
            result[f"{col}_z"] = np.nan
            continue

        # Forward-fill only. This preserves pre-first-observation missing history.
        stable = raw.ffill()
        roll_mean = stable.rolling(window=lookback, min_periods=max(12, lookback // 10)).mean()
        roll_std = stable.rolling(window=lookback, min_periods=max(12, lookback // 10)).std().replace(0, np.nan)
        zscore = ((stable - roll_mean) / roll_std).replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0)

        presence = raw.notna().astype(float)
        weighted_sum += zscore.fillna(0.0) * float(weight) * presence
        available_weight += abs(float(weight)) * presence
        result[f"{col}_z"] = zscore
        result[col] = raw

    result["macro_coverage_ratio"] = (available_weight / max(total_abs_weight, 1e-9)).clip(0.0, 1.0)
    result["macro_regime_index"] = (
        weighted_sum / available_weight.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
    result["macro_regime_shock"] = result["macro_regime_index"].diff().fillna(0.0).clip(-5.0, 5.0)

    if "macro_directional_flag" in result.columns:
        upstream_raw = result["macro_directional_flag"]
    else:
        upstream_raw = pd.Series(0.0, index=result.index, dtype=float)
    upstream_flag = pd.to_numeric(upstream_raw, errors="coerce").fillna(0.0)
    derived_flag = np.select(
        [result["macro_regime_shock"] >= 0.15, result["macro_regime_shock"] <= -0.15],
        [1, -1],
        default=0,
    )
    result["macro_directional_flag"] = np.where(upstream_flag != 0.0, upstream_flag, derived_flag).astype(float)
    return result

def engineer_features(df: pd.DataFrame, is_forex: bool = False) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    Combines various indicators commonly used in predicting price action.
    
    Args:
        df: Input DataFrame with at least 'close' column, sorted by time.
        is_forex: If true, use shorter rolling windows for faster adaptation.
        
    Returns:
        DataFrame transformed with all features.
    """
    df = df.copy()
    
    # Sort just in case
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
    # Lags
    df = add_lag_features(df, target_col='close', lags=5)
    
    # Rolling stats - shorter for forex to avoid dropping too much data during walk-forward
    windows = [4, 7] if is_forex else [7, 14]
    df = add_rolling_features(df, target_col='close', windows=windows)
    
    # RSI
    df = add_rsi(df, target_col='close', period=14)
    
    # MACD
    df = add_macd(df, target_col='close')

    # Macro regime composite (for exogenous context)
    # Use shorter lookback for forex/hourly dynamics.
    macro_lookback = 60 if is_forex else 120
    df = add_macro_regime_features(df, lookback=macro_lookback)
    
    return df
