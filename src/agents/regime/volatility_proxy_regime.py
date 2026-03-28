import numpy as np
import pandas as pd


def compute_volatility_proxy_regime(
    close_prices: pd.Series,
    vol_window: int = 20,
    low_percentile: float = 33.33,
    high_percentile: float = 66.67,
) -> pd.Series:
    """
    Computes a simple 3-state volatility-proxy regime based on rolling standard deviation percentiles.
    
    This is intended as a modular utility for data-starved symbols where HMM fitting fails.
    Returns a categorical series with values in [0, 1, 2] corresponding to LOW_VOL, MID_VOL, HIGH_VOL.
    
    Parameters
    ----------
    close_prices : pd.Series
        Historical close prices.
    vol_window : int
        Rolling window for price standard deviation.
    low_percentile : float
        Percentile boundary for LOW_VOL vs MID_VOL.
    high_percentile : float
        Percentile boundary for MID_VOL vs HIGH_VOL.
        
    Returns
    -------
    pd.Series
        Regime state series (0=LOW, 1=MID, 2=HIGH).
    """
    # Use log returns for true volatility representation
    returns = np.log(close_prices / close_prices.shift(1))
    
    # Rolling standard deviation as volatility proxy
    rolling_vol = returns.rolling(window=vol_window, min_periods=min(10, vol_window)).std()
    
    # Calculate expanding percentiles to prevent future leakage
    expanding_p33 = rolling_vol.expanding().quantile(low_percentile / 100.0)
    expanding_p66 = rolling_vol.expanding().quantile(high_percentile / 100.0)
    
    # Assign states: 0=LOW, 1=MID, 2=HIGH
    states = pd.Series(1, index=close_prices.index, name="proxy_regime")
    states[rolling_vol <= expanding_p33] = 0
    states[rolling_vol >= expanding_p66] = 2
    
    # Forward fill early NaNs if any, defaulting to mid
    states = states.fillna(1).astype(int)
    
    return states
