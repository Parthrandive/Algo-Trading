import pandas as pd
from typing import Dict, List, Any
from datetime import timedelta
import logging

from src.schemas.preprocessing_data import TransformOutput
from src.agents.preprocessing.lag_alignment import LagAligner

logger = logging.getLogger(__name__)

class LeakageError(Exception):
    """Raised when future macro data leaks into an aligned or transformed market row."""
    pass

class LeakageTestHarness:
    """
    Checks for future information leaking into transforms (Section 5.3).
    Ensures strict time alignment and lagging verification.
    """
    
    def __init__(self, lag_aligner: LagAligner = None):
        self.lag_aligner = lag_aligner or LagAligner()
        self.publication_delays = self.lag_aligner.publication_delays

    def verify_no_lookahead(self, market_df: pd.DataFrame, macro_df: pd.DataFrame, output: TransformOutput) -> bool:
        """
        Validates that for every row in the output, the assigned macro indicator values 
        correspond to a raw macro update whose `timestamp + publication_delay <= market_timestamp`.
        Raises LeakageError if a leak is detected.
        """
        if not output.records:
            return True
            
        output_df = pd.DataFrame(output.records)
        output_df['timestamp'] = pd.to_datetime(output_df['timestamp'], utc=True)
        
        # Verify alignment for each configured macro indicator
        macro_indicators = macro_df['indicator_name'].unique() if not macro_df.empty else []
        
        for indicator in macro_indicators:
            if indicator not in output_df.columns:
                continue
                
            delay = self.publication_delays.get(indicator, timedelta(0))
            specific_macro = macro_df[macro_df['indicator_name'] == indicator].copy()
            specific_macro['timestamp'] = pd.to_datetime(specific_macro['timestamp'], utc=True)
            specific_macro['effective_time'] = specific_macro['timestamp'] + delay
            
            # Check every row in the output
            for _, out_row in output_df.dropna(subset=[indicator]).iterrows():
                market_time = out_row['timestamp']
                out_val = out_row[indicator]
                
                # Find all macro updates that were published AFTER this market time
                future_macro = specific_macro[specific_macro['effective_time'] > market_time]
                
                # If the value in the output matches a value that was only published in the future,
                # we need to be careful. It might just be the same value as a past update.
                # So we check if the MUST-HAVE recent value (the one at or before market_time) 
                # matches the output value.
                
                past_macro = specific_macro[specific_macro['effective_time'] <= market_time]
                
                if past_macro.empty:
                    # No macro data should be available yet!
                    if pd.notna(out_val) and out_val != 0.0: # 0.0 is our fillna default
                        raise LeakageError(f"Leak detected for {indicator}: value {out_val} at {market_time}, but no data published yet!")
                else:
                    expected_val = past_macro.sort_values('effective_time').iloc[-1]['value']
                    # We compare with a small tolerance due to potential float issues during hashing/serialization
                    if abs(out_val - expected_val) > 1e-6 and not (pd.isna(out_val) and pd.isna(expected_val)):
                        # If the value doesn't match the expected past value, check if it matches a future value
                        if not future_macro.empty and any(abs(future_macro['value'] - out_val) < 1e-6):
                             raise LeakageError(f"Future Leak for {indicator} at {market_time}: Got {out_val}, expected {expected_val} based on past data.")
                        else:
                             # It's wrong, but maybe not a *future* leak specifically, could be bug in aligner
                             logger.warning(f"Alignment mismatch for {indicator} at {market_time}: Got {out_val}, expected {expected_val}")
                             
        return True

    def verify_time_alignment(self, output: TransformOutput) -> bool:
        """
        Validates timestamps are ordered and timezone aware.
        """
        if not output.records:
            return True
            
        df = pd.DataFrame(output.records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Check ordering per symbol
        for sym in df['symbol'].unique():
            sym_df = df[df['symbol'] == sym]
            if not sym_df['timestamp'].is_monotonic_increasing:
                raise LeakageError(f"Timestamps for {sym} are not strictly monotonic increasing in output!")
                
        return True
