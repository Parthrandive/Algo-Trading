import numpy as np
import pandas as pd

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

DAILY_FEATURE_COLUMNS = [
    "daily_ma_20",
    "daily_ma_50",
    "daily_ma_200",
    "daily_rsi",
    "daily_bollinger_upper",
    "daily_bollinger_lower",
    "daily_bollinger_width",
    "daily_adx",
    "daily_trend_strength",
    "daily_trend_bullish",
]


def apply_daily_trend_confirmation(
    pred_labels: np.ndarray,
    probs: np.ndarray,
    daily_trend_bullish: np.ndarray,
    *,
    up_label: int = 2,
    neutral_label: int = 1,
    mode: str = "hard",
    up_penalty: float = 0.5,
    soft_confidence_cut: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforce daily trend confirmation on directional long signals.

    Rule:
    - hard: If predicted label is `up_label` and daily trend is not bullish,
      downgrade to `neutral_label`.
    - soft: Penalize UP probability first; neutralize only if adjusted UP
      confidence drops below `soft_confidence_cut`.
    """
    labels = np.asarray(pred_labels, dtype=np.int64).copy()
    adjusted_probs = np.asarray(probs, dtype=np.float64).copy()
    trend = np.asarray(daily_trend_bullish, dtype=np.float64)

    if len(labels) == 0 or len(trend) != len(labels):
        return labels, adjusted_probs, np.zeros(len(labels), dtype=bool)

    bullish = trend >= 0.5
    candidate_mask = (labels == int(up_label)) & (~bullish)
    if not candidate_mask.any():
        return labels, adjusted_probs, candidate_mask

    mode_normalized = str(mode).strip().lower()
    if mode_normalized not in {"hard", "soft"}:
        raise ValueError(f"Unsupported daily trend confirmation mode: {mode}")

    if adjusted_probs.ndim == 2 and adjusted_probs.shape[1] >= 3:
        up_idx = int(up_label)
        neutral_idx = int(neutral_label)

        if mode_normalized == "hard":
            up_prob = adjusted_probs[candidate_mask, up_idx].copy()
            adjusted_probs[candidate_mask, neutral_idx] = np.clip(
                adjusted_probs[candidate_mask, neutral_idx] + up_prob,
                0.0,
                1.0,
            )
            adjusted_probs[candidate_mask, up_idx] = 0.0
            row_sum = adjusted_probs[candidate_mask].sum(axis=1, keepdims=True)
            adjusted_probs[candidate_mask] = np.divide(
                adjusted_probs[candidate_mask],
                np.where(row_sum == 0.0, 1.0, row_sum),
            )
            labels[candidate_mask] = int(neutral_label)
            return labels, adjusted_probs, candidate_mask

        clipped_penalty = float(np.clip(up_penalty, 0.0, 1.0))
        removed = adjusted_probs[candidate_mask, up_idx] * clipped_penalty
        adjusted_probs[candidate_mask, up_idx] = np.clip(
            adjusted_probs[candidate_mask, up_idx] - removed,
            0.0,
            1.0,
        )
        adjusted_probs[candidate_mask, neutral_idx] = np.clip(
            adjusted_probs[candidate_mask, neutral_idx] + removed,
            0.0,
            1.0,
        )
        row_sum = adjusted_probs[candidate_mask].sum(axis=1, keepdims=True)
        adjusted_probs[candidate_mask] = np.divide(
            adjusted_probs[candidate_mask],
            np.where(row_sum == 0.0, 1.0, row_sum),
        )
        adjusted_up_conf = adjusted_probs[candidate_mask, up_idx]
        soft_neutralized = np.zeros(len(labels), dtype=bool)
        soft_neutralized[candidate_mask] = adjusted_up_conf < float(soft_confidence_cut)
        labels[soft_neutralized] = int(neutral_label)
        return labels, adjusted_probs, soft_neutralized

    if mode_normalized == "hard":
        labels[candidate_mask] = int(neutral_label)
        return labels, adjusted_probs, candidate_mask

    # Soft mode without probabilities cannot penalize confidence; fallback to label-only no-op.
    return labels, adjusted_probs, np.zeros(len(labels), dtype=bool)


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


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=high.index, dtype=float)

    tr_components = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()

    plus_di = 100.0 * plus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx


def _compute_daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=["availability_timestamp", "open", "high", "low", "close", "volume"])

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return pd.DataFrame(columns=["availability_timestamp", "open", "high", "low", "close", "volume"])

    for column in ["open", "high", "low", "close", "volume"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    grouped = (
        frame.set_index("timestamp")
        .resample("1D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    if grouped.empty:
        return pd.DataFrame(columns=["availability_timestamp", "open", "high", "low", "close", "volume"])

    # Daily candle t becomes usable at the start of t+1. This prevents same-day look-ahead.
    grouped["availability_timestamp"] = grouped["timestamp"] + pd.Timedelta(days=1)
    return grouped[["availability_timestamp", "open", "high", "low", "close", "volume"]]


def _compute_daily_feature_frame(daily_ohlcv: pd.DataFrame) -> pd.DataFrame:
    if daily_ohlcv.empty:
        return pd.DataFrame(columns=["availability_timestamp", *DAILY_FEATURE_COLUMNS])

    out = daily_ohlcv.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    out["daily_ma_20"] = close.rolling(window=20, min_periods=20).mean()
    out["daily_ma_50"] = close.rolling(window=50, min_periods=50).mean()
    out["daily_ma_200"] = close.rolling(window=200, min_periods=200).mean()

    rsi_df = add_rsi(pd.DataFrame({"close": close}), target_col="close", period=14)
    out["daily_rsi"] = rsi_df["rsi"]

    roll_std = close.rolling(window=20, min_periods=20).std()
    out["daily_bollinger_upper"] = out["daily_ma_20"] + (2.0 * roll_std)
    out["daily_bollinger_lower"] = out["daily_ma_20"] - (2.0 * roll_std)
    out["daily_bollinger_width"] = (
        (out["daily_bollinger_upper"] - out["daily_bollinger_lower"]) / out["daily_ma_20"].replace(0, np.nan)
    )

    out["daily_adx"] = _compute_adx(out["high"], out["low"], out["close"], period=14)
    out["daily_trend_strength"] = close / out["daily_ma_50"].replace(0, np.nan) - 1.0
    bullish = (close > out["daily_ma_50"]) & (out["daily_ma_20"] > out["daily_ma_50"])
    out["daily_trend_bullish"] = bullish.astype(float)
    out.loc[out["daily_ma_50"].isna() | out["daily_ma_20"].isna(), "daily_trend_bullish"] = np.nan

    return out[["availability_timestamp", *DAILY_FEATURE_COLUMNS]]


def add_daily_timeframe_features(
    df: pd.DataFrame,
    daily_reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adds daily indicators and aligns them onto intraday timestamps via as-of join.

    Leakage guardrail:
    - Daily candle for date t is only available from t+1 (availability_timestamp),
      so intraday rows on t never consume same-day final close-derived values.
    """
    if "timestamp" not in df.columns:
        result = df.copy()
        for column in DAILY_FEATURE_COLUMNS:
            if column not in result.columns:
                result[column] = np.nan
        return result

    target = df.copy()
    target["_row_order"] = np.arange(len(target))
    target["timestamp"] = pd.to_datetime(target["timestamp"], utc=True, errors="coerce")
    target = target.sort_values("timestamp")

    daily_input = daily_reference_df if daily_reference_df is not None else df
    daily_ohlcv = _compute_daily_ohlcv(daily_input)
    daily_features = _compute_daily_feature_frame(daily_ohlcv)

    if daily_features.empty:
        for column in DAILY_FEATURE_COLUMNS:
            target[column] = np.nan
        target = target.sort_values("_row_order").drop(columns=["_row_order"])
        return target

    merged = pd.merge_asof(
        target,
        daily_features.sort_values("availability_timestamp"),
        left_on="timestamp",
        right_on="availability_timestamp",
        direction="backward",
    )
    merged = merged.drop(columns=["availability_timestamp"], errors="ignore")
    merged = merged.sort_values("_row_order").drop(columns=["_row_order"])
    return merged


def engineer_features(
    df: pd.DataFrame,
    is_forex: bool = False,
    *,
    include_daily_features: bool = True,
    daily_reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    Combines various indicators commonly used in predicting price action.
    
    Args:
        df: Input DataFrame with at least 'close' column, sorted by time.
        is_forex: If true, use shorter rolling windows for faster adaptation.
        include_daily_features: Enables daily indicator fusion onto intraday rows.
        daily_reference_df: Optional external daily/intraday frame used to compute
            daily indicators before as-of alignment.
        
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

    if include_daily_features:
        df = add_daily_timeframe_features(df, daily_reference_df=daily_reference_df)
    
    return df
