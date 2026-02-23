# Source Inventory v0.1

| Source Name | Type | Key Data | Licensing Status | Fallback Rank |
| :--- | :--- | :--- | :--- | :--- |
| **Official NSE API** | API | Real-time Price, Volume, Corp Actions | **Primary** (Official/Paid) | 1 |
| **Broker API (e.g., Zerodha/Upstox)** | API | Real-time Price, Orders, Holdings | **Secondary** (Broker Account) | 2 |
| **NSEPython** | Library (Wrapper) | Option Chain, Historical Data | **Open Source** (Unofficial) | 3 |
| **Yahoo Finance (yfinance)** | Library (Wrapper) | Historical OHLCV (Validation) | **Open Source** (Unofficial) | 4 |
| **RBI Website** | Web | Interest Rates, FX Reserves | **Public Domain** | Macro-1 |
| **Ministry of Commerce** | Web | WPI, CPI, Trade Balance | **Public Domain** | Macro-1 |
| **AMFI / SEBI** | Web | FII/DII Flows | **Public Domain** | Macro-1 |
| **Exchange Scraping** | Web Scraper | Corp Actions, Circuit Limits | **Fallback Only** (High Risk) | 5 |

## Notes
- **Usage Policy**: Unpublished Price Sensitive Information (UPSI) is strictly prohibited. Only publicly available or officially licensed data is to be used.
- **Redundancy**: Broker APIs are preferred for execution-critical paths if the Official NSE API has latency/cost issues, but Official API is the "Gold Standard" for data integrity.
- **Fallback**: Scraping is a last resort and must be rate-limited and compliant with `robots.txt`.
- **Macro Proxy Governance**: Proxy inclusion/exclusion rationale is tracked in `docs/governance/macro_proxy_justification_v1.md` (Brent included, DXY excluded for Week 3 catalog).
