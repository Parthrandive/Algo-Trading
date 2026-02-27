"""
Macro Monitor — concrete macro source clients package.

Each module exposes ONE client class responsible for a single logical source:

    MOSPIClient        — CPI, WPI, IIP  (mospi_client.py)
    RBIClient          — FX Reserves, RBI Bulletins, India 10Y  (rbi_client.py)
    NSEDIIFIIClient    — FII Flow, DII Flow  (nse_fiidii_client.py)
    BondSpreadClient   — INDIA_US_10Y_SPREAD (computed from India + US legs)  (bond_spread_client.py)
"""
