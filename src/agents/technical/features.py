import pandas as pd
import numpy as np

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

    rs = avg_gain / avg_loss
    result['rsi'] = 100 - (100 / (1 + rs))
    # Handle edge case where avg_loss is 0
    result['rsi'] = result['rsi'].fillna(100)
    # Mask initial NaNs correctly depending on the rolling window
    result.loc[result.index[:period-1], 'rsi'] = np.nan
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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    Combines various indicators commonly used in predicting price action.
    
    Args:
        df: Input DataFrame with at least 'close' column, sorted by time.
        
    Returns:
        DataFrame transformed with all features.
    """
    df = df.copy()
    
    # Sort just in case
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        
    # Lags
    df = add_lag_features(df, target_col='close', lags=5)
    
    # Rolling stats
    df = add_rolling_features(df, target_col='close', windows=[7, 14])
    
    # RSI
    df = add_rsi(df, target_col='close', period=14)
    
    # MACD
    df = add_macd(df, target_col='close')
    
    return df
