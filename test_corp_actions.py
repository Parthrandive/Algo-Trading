import yfinance as yf
import nsepython

print("--- YFinance ---")
t = yf.Ticker("RELIANCE.NS")
print(t.actions.tail())

print("--- NSEPython ---")
try:
    # Not sure if nsepython has a direct corp action function exposed, let's try some common names
    # nse_fiixed_income, nse_holiday etc.
    # What about looking up the module dir?
    print("Methods available in nsepython related to corp:")
    corp_methods = [m for m in dir(nsepython) if 'corp' in m.lower() or 'event' in m.lower() or 'dividend' in m.lower()]
    print(corp_methods)
except Exception as e:
    print(e)
