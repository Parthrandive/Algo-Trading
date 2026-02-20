
import nsepython

def test_nsepython():
    print("Testing nsepython...")
    try:
        # transformative function? nsepython usually has nse_quote or something
        # Let's try to print help or dir if we don't know the exact API
        # but usually it's nse_quote("RELIANCE")
        q = nsepython.nse_quote("RELIANCE")
        print(f"Quote for RELIANCE: {q}")
        
        print("Testing historical data...")
        # equity_history(symbol, series, start_date, end_date)
        # dates in dd-mm-yyyy
        h = nsepython.equity_history("RELIANCE", "EQ", "01-01-2024", "10-01-2024")
        print(f"History head: {h[0] if len(h) > 0 else 'Empty'}")
        print(f"Fetched {len(h)} records.")
    except Exception as e:
        print(f"Error calling nsepython: {e}")

if __name__ == "__main__":
    test_nsepython()
