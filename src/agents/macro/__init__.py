"""
Macro Monitor Agent
===================
Ingests India macro and cross-asset indicators with scheduled jobs,
freshness SLAs, and alerting.

Package layout
--------------
client.py       — MacroClientInterface Protocol + DateRange
clients/        — One concrete stub per data source
recorder.py     — MacroSilverRecorder (Parquet + optional DB)
"""
