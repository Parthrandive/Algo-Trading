from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.strategic.artifacts import resolve_code_hash
from src.agents.strategic.config import OBSERVATION_SCHEMA_VERSION
from src.agents.strategic.observation import ObservationAssembler
from src.agents.strategic.policies import PPOPolicyFoundation, SACPolicyFoundation, TD3PolicyFoundation
from src.agents.strategic.schemas import StrategicObservation
from src.db.phase3_recorder import Phase3Recorder
from src.db.queries import get_bars

POLICY_FACTORIES = {
    "SAC": SACPolicyFoundation,
    "PPO": PPOPolicyFoundation,
    "TD3": TD3PolicyFoundation,
}


def parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_algorithms(value: str) -> list[str]:
    items = [item.strip().upper() for item in value.split(",") if item.strip()]
    deduped: list[str] = []
    for item in items:
        if item not in POLICY_FACTORIES:
            raise ValueError(f"Unsupported algorithm: {item}. Supported: {sorted(POLICY_FACTORIES)}")
        if item not in deduped:
            deduped.append(item)
    if not deduped:
        raise ValueError("No algorithms provided.")
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train offline SAC/PPO/TD3 teacher policies on Phase 3 observations.")
    parser.add_argument("--symbol", required=True, help="Target symbol (for example RELIANCE.NS)")
    parser.add_argument("--start", required=True, help="ISO8601 start timestamp")
    parser.add_argument("--end", required=True, help="ISO8601 end timestamp")
    parser.add_argument("--interval", default="1h", help="OHLCV interval used for reward prices")
    parser.add_argument("--algorithms", default="SAC,PPO,TD3", help="Comma-separated algorithms")
    parser.add_argument("--reward-name", default="ra_drl_composite", choices=("step_return", "ra_drl_composite"))
    parser.add_argument("--total-timesteps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--database-url", default=None, help="Optional SQLAlchemy DB URL override")
    parser.add_argument("--output-dir", default="data/models/strategic/offline_runs")
    parser.add_argument("--report-dir", default="data/reports/phase3/offline_training")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--min-points", type=int, default=256)
    parser.add_argument("--skip-db-write", action="store_true", help="Train and write artifacts, but skip DB persistence")
    return parser


def build_training_dataset(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    database_url: str | None,
) -> tuple[list[Any], list[float], dict[str, Any]]:
    phase3_reader = Phase3Recorder(database_url=database_url, bootstrap_schema=False)
    assembler = ObservationAssembler(database_url=database_url, recorder=phase3_reader)
    bars = get_bars(symbol=symbol, start=start, end=end, interval=interval)
    bars = bars.sort_values("timestamp")
    if "close" not in bars.columns:
        bars = pd.DataFrame()

    metadata: dict[str, Any] = {
        "symbol": symbol,
        "price_rows": int(bars.shape[0]),
        "interval": interval,
        "price_source": "ohlcv",
        "observation_source": "phase2_bundle",
    }

    if bars.empty:
        raise ValueError(f"No OHLCV bars found for {symbol} ({interval}) between {start} and {end}.")

    try:
        observations = assembler.build_symbol_observations(symbol=symbol, start=start, end=end)
    except Exception:
        observations = []
    metadata["observation_rows"] = len(observations)

    if len(observations) < 2:
        fallback_observations, fallback_prices = _build_observation_fallback_from_bars(symbol=symbol, bars=bars)
        metadata["observation_source"] = "ohlcv_fallback"
        metadata["observation_rows"] = len(fallback_observations)
        metadata["price_alignment_rows"] = len(fallback_prices)
        return fallback_observations, fallback_prices, metadata

    obs_frame = pd.DataFrame(
        {
            "obs_idx": list(range(len(observations))),
            "timestamp": [item.timestamp for item in observations],
        }
    )
    obs_frame["timestamp"] = pd.to_datetime(obs_frame["timestamp"], utc=True, errors="coerce")
    price_frame = bars[["timestamp", "close"]].copy()
    price_frame["timestamp"] = pd.to_datetime(price_frame["timestamp"], utc=True, errors="coerce")
    price_frame = price_frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    aligned = pd.merge_asof(
        obs_frame.sort_values("timestamp"),
        price_frame,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("6h"),
    ).dropna(subset=["close"])

    if len(aligned) < 2:
        fallback_observations, fallback_prices = _build_observation_fallback_from_bars(symbol=symbol, bars=bars)
        metadata["observation_source"] = "ohlcv_fallback"
        metadata["price_source"] = "ohlcv"
        metadata["observation_rows"] = len(fallback_observations)
        metadata["price_alignment_rows"] = len(fallback_prices)
        return fallback_observations, fallback_prices, metadata

    selected_obs = [observations[int(idx)] for idx in aligned["obs_idx"].astype(int).tolist()]
    prices = aligned["close"].astype(float).tolist()
    metadata["price_alignment_rows"] = int(len(selected_obs))
    return selected_obs, prices, metadata


def _build_observation_fallback_from_bars(
    *,
    symbol: str,
    bars: pd.DataFrame,
) -> tuple[list[StrategicObservation], list[float]]:
    frame = bars[["timestamp", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError(f"Insufficient OHLCV rows to build fallback observations for {symbol}.")

    returns = frame["close"].pct_change().fillna(0.0)
    rolling_vol = returns.rolling(window=24, min_periods=3).std().fillna(returns.std(ddof=0) or 1e-6)
    baseline_vol = rolling_vol.rolling(window=120, min_periods=12).median().fillna(rolling_vol.median() or 1e-6)
    baseline_vol = baseline_vol.replace(0.0, max(float(rolling_vol.mean()), 1e-6))
    sentiment_z = (returns - returns.rolling(window=48, min_periods=8).mean()) / (
        returns.rolling(window=48, min_periods=8).std().replace(0.0, np.nan)
    )
    sentiment_z = sentiment_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    observations: list[StrategicObservation] = []
    prices = frame["close"].astype(float).tolist()
    for idx, row in frame.iterrows():
        close = float(row["close"])
        ts = row["timestamp"].to_pydatetime()
        step_return = float(returns.iloc[idx])
        vol = max(float(rolling_vol.iloc[idx]), 1e-6)
        base = max(float(baseline_vol.iloc[idx]), 1e-6)
        vol_ratio = vol / base
        direction = "up" if step_return > 5e-4 else "down" if step_return < -5e-4 else "neutral"
        consensus_direction = "BUY" if direction == "up" else "SELL" if direction == "down" else "HOLD"
        if vol_ratio > 1.5:
            regime_state = "Bear" if step_return < 0.0 else "Bull"
        elif abs(step_return) < 3e-4:
            regime_state = "Sideways"
        else:
            regime_state = "Bull" if step_return >= 0.0 else "Bear"
        confidence = float(np.clip(abs(step_return) / (vol + 1e-6), 0.05, 0.95))
        transition_prob = float(np.clip(vol_ratio / 2.0, 0.0, 1.0))
        z_score = float(np.clip(sentiment_z.iloc[idx], -3.0, 3.0))
        observations.append(
            StrategicObservation(
                timestamp=ts,
                symbol=symbol,
                snapshot_id=f"{symbol}:{ts.isoformat()}:{OBSERVATION_SCHEMA_VERSION}:fallback",
                technical_direction=direction,
                technical_confidence=confidence,
                price_forecast=close,
                var_95=float(-1.65 * vol),
                es_95=float(-2.33 * vol),
                regime_state=regime_state,
                regime_transition_prob=transition_prob,
                sentiment_score=float(np.clip(z_score / 3.0, -1.0, 1.0)),
                sentiment_z_t=z_score,
                consensus_direction=consensus_direction,
                consensus_confidence=confidence,
                crisis_mode=bool(vol_ratio > 2.0),
                agent_divergence=False,
                quality_status="warn",
            )
        )
    return observations, prices


def main() -> int:
    args = build_parser().parse_args()
    start = parse_datetime(args.start)
    end = parse_datetime(args.end)
    if end <= start:
        raise ValueError("end must be greater than start")

    algorithms = parse_algorithms(args.algorithms)
    run_id = args.run_id or f"phase3_offline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_output_dir = Path(args.output_dir) / run_id
    run_report_dir = Path(args.report_dir)
    run_report_dir.mkdir(parents=True, exist_ok=True)

    observations, prices, dataset_meta = build_training_dataset(
        symbol=args.symbol,
        start=start,
        end=end,
        interval=args.interval,
        database_url=args.database_url,
    )
    if len(observations) < args.min_points:
        raise ValueError(f"Insufficient training points: {len(observations)} < min_points ({args.min_points})")
    if len(prices) != len(observations):
        raise ValueError("prices and observations must have matching lengths")

    recorder = None if args.skip_db_write else Phase3Recorder(database_url=args.database_url)
    code_hash = resolve_code_hash()

    results_payload: list[dict[str, Any]] = []
    for algo in algorithms:
        trainer = POLICY_FACTORIES[algo]()
        result = trainer.train_offline(
            observations=observations,
            prices=prices,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            reward_name=args.reward_name,
            output_dir=run_output_dir,
            device=args.device,
        )
        if recorder is not None:
            recorder.save_rl_policy(
                result.as_policy_row(observation_schema_version=OBSERVATION_SCHEMA_VERSION)
            )
            recorder.save_rl_training_run(
                result.as_training_run_row(dataset_snapshot_id=run_id, code_hash=code_hash)
            )
        results_payload.append(
            {
                "algorithm": algo,
                "policy_id": result.policy_id,
                "checkpoint_path": str(result.checkpoint_path),
                "episodes": result.episodes,
                "total_steps": result.total_steps,
                "duration_seconds": result.duration_seconds,
                "metrics": result.metrics,
                "params": result.params,
            }
        )

    summary = {
        "run_id": run_id,
        "symbol": args.symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "reward_name": args.reward_name,
        "dataset": dataset_meta,
        "algorithms": algorithms,
        "results": results_payload,
        "code_hash": code_hash,
        "db_write_enabled": not args.skip_db_write,
    }
    summary_path = run_report_dir / f"{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "summary_path": str(summary_path), "results": results_payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
