# Data Scale Roadmap - End of Phase 1

This roadmap finalizes the volume checkpoints and tier migration triggers for data processing capability limits defined in the Phase 1 strategy.

## Current Baselines (Daily Volumes)
- **Sentinel (Tick + OHLCV)**: ~14 GB Bronze raw payloads. ~2.1 GB Silver structured data.
- **Macro Monitor**: < 50 MB / day.
- **Preprocessing (Feature Vectors)**: ~1 GB output.
- **Textual Logs / Meta**: ~5 GB raw text output -> < 1 GB DB structured output.

## Volume Checkpoints
1. **At 50 GB / Day (Expected Mid-Phase 2)**: DB Write batch sizes must increase, shifting to larger queue brokers and streaming inserts instead of parallel direct inserts.
2. **At 200 GB / Day (Expected Phase 3)**: Time-series database replacement mandatory for Gold-tier. Read queries on current layout will timeout.
3. **At 1 TB / Day (Long Term)**: Bronze storage transitions from daily partition to hourly partitions to limit query scanning costs.

## Tier Migration Triggers
- Raw JSON logs older than 7 days will push from hot disks to cold S3 object storage (Bronze deep tier).
- Silver Parquet dumps persist directly into DB ingestion queue and archive automatically after 30 days. Feature lookup occurs in Gold KV cache.
