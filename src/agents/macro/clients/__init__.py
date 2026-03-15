"""
Macro Monitor — concrete macro source clients package.

Each module exposes ONE client class responsible for a single logical source:

    AkShareClient      — REPO_RATE, US_10Y (akshare-backed)  (akshare_client.py)
    MOSPIClient        — CPI, WPI, IIP  (mospi_client.py)
    RBIClient          — FX Reserves, RBI Bulletins, India 10Y  (rbi_client.py)
    NSEDIIFIIClient    — FII Flow, DII Flow  (nse_fiidii_client.py)
    BondSpreadClient   — INDIA_US_10Y_SPREAD (computed from India + US legs)  (bond_spread_client.py)
"""

from src.agents.macro.clients.akshare_client import AkShareClient
from src.agents.macro.clients.bond_spread_client import BondSpreadClient
from src.agents.macro.clients.fx_reserves_client import FXReservesClient
from src.agents.macro.clients.mospi_client import MOSPIClient
from src.agents.macro.clients.nse_fiidii_client import NSEDIIFIIClient
from src.agents.macro.clients.rbi_client import RBIClient

__all__ = [
    "AkShareClient",
    "MOSPIClient",
    "RBIClient",
    "NSEDIIFIIClient",
    "FXReservesClient",
    "BondSpreadClient",
]
