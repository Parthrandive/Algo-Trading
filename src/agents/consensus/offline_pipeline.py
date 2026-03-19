from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from src.agents.regime.models.hmm_regime import HMMRegimeModel
from src.agents.regime.models.ood_detector import OODDetector
from src.agents.regime.models.pearl_meta import PearlMetaModel
from src.agents.sentiment.sentiment_agent import SentimentAgent
from src.agents.technical.features import engineer_features


MACRO_COLUMNS = [
    "CPI",
    "WPI",
    "IIP",
    "FII_FLOW",
    "DII_FLOW",
    "FX_RESERVES",
    "INDIA_US_10Y_SPREAD",
    "RBI_BULLETIN",
    "REPO_RATE",
    "US_10Y",
]

CLASS_TO_INT = {"down": -1, "neutral": 0, "up": 1}
INT_TO_CLASS = {-1: "down", 0: "neutral", 1: "up"}
REGIME_TO_SCORE = {
    "Bull": 0.85,
    "Bear": -0.85,
    "Sideways": 0.0,
    "Crisis": -0.6,
    "RBI-Band transition": -0.2,
    "Alien": -1.0,
}
REGIME_TO_PROTECTIVE = {
    "Bull": False,
    "Bear": True,
    "Sideways": False,
    "Crisis": True,
    "RBI-Band transition": True,
    "Alien": True,
}
RISK_LEVEL_TO_FLOAT = {"full_risk": 0.0, "reduced_risk": 0.5, "neutral_cash": 1.0}


@dataclass
class PipelineConfig:
    symbols: list[str]
    output_root: Path
    silver_ohlcv_root: Path = Path("data/silver/ohlcv")
    silver_macro_root: Path = Path("data/silver/macro")
    textual_canonical_path: Path = Path("docs/reports/day4_sync_s2/artifacts/textual_canonical_2026-03-05.parquet")
    textual_sidecar_path: Path = Path("docs/reports/day4_sync_s2/artifacts/textual_sidecar_2026-03-05.parquet")
    min_train_rows: int = 60
    test_size_ratio: float = 0.2
    tx_cost_bps: float = 8.0
    neutral_vol_scale: float = 0.5
    sentiment_lookback_hours: int = 24
    sentiment_stale_hours: int = 6
    seed: int = 42
    skip_challenger: bool = False


@dataclass
class TechnicalModelBundle:
    scaler: StandardScaler
    cls_model: LogisticRegression | None
    reg_model: Ridge | None
    feature_columns: list[str]
    class_prior: dict[int, float]
    fallback_mean_return: float


@dataclass
class RegimeModelBundle:
    hmm: HMMRegimeModel | None
    pearl: PearlMetaModel | None
    ood: OODDetector | None
    fitted: bool


@dataclass
class ChallengerModel:
    low_model: BayesianRidge
    high_model: BayesianRidge
    feature_columns: list[str]
    residual_std: float
    transition_switch: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("=", "_").replace(".", "_")


def _load_symbol_ohlcv(symbol: str, root: Path) -> pd.DataFrame:
    symbol_dir = root / symbol
    if not symbol_dir.exists():
        return pd.DataFrame()
    files = sorted(symbol_dir.glob("**/*.parquet"))
    if not files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for file_path in files:
        try:
            frames.append(pd.read_parquet(file_path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if "interval" in df.columns:
        df = df[df["interval"].astype(str) == "1h"]
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df["symbol"] = symbol

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0.0)
    return df


def _load_macro_frame(root: Path) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame(columns=["timestamp", "indicator_name", "value"])

    rows: list[pd.DataFrame] = []
    for indicator_dir in sorted(root.glob("*")):
        if not indicator_dir.is_dir():
            continue
        for file_path in sorted(indicator_dir.glob("**/*.parquet")):
            try:
                part = pd.read_parquet(file_path)
            except Exception:
                continue
            if "timestamp" not in part.columns or "value" not in part.columns:
                continue
            if "indicator_name" not in part.columns:
                part["indicator_name"] = indicator_dir.name
            rows.append(part[["timestamp", "indicator_name", "value"]].copy())

    if not rows:
        return pd.DataFrame(columns=["timestamp", "indicator_name", "value"])

    macro = pd.concat(rows, ignore_index=True)
    macro["timestamp"] = pd.to_datetime(macro["timestamp"], utc=True, errors="coerce")
    macro["indicator_name"] = macro["indicator_name"].astype(str).str.upper().str.strip()
    macro["value"] = pd.to_numeric(macro["value"], errors="coerce")
    macro = macro.dropna(subset=["timestamp", "indicator_name", "value"])
    macro = macro.sort_values(["indicator_name", "timestamp"]).drop_duplicates(
        subset=["indicator_name", "timestamp"],
        keep="last",
    )
    return macro.reset_index(drop=True)


def merge_macro_asof(
    market_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = market_df.sort_values("timestamp").reset_index(drop=True).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    if macro_df.empty:
        for col in MACRO_COLUMNS:
            out[col] = np.nan
        return out, {"macro_rows": 0, "asof_safe_violations": 0}

    pivot = (
        macro_df.pivot_table(index="timestamp", columns="indicator_name", values="value", aggfunc="last")
        .sort_index()
        .reset_index()
    )
    pivot["timestamp"] = pd.to_datetime(pivot["timestamp"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    pivot = pivot.dropna(subset=["timestamp"]).reset_index(drop=True)
    merged = pd.merge_asof(
        out,
        pivot,
        on="timestamp",
        direction="backward",
    )

    for col in MACRO_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    leak_check = 0
    for col in MACRO_COLUMNS:
        series = macro_df[macro_df["indicator_name"] == col][["timestamp", "value"]].sort_values("timestamp")
        if series.empty:
            continue
        series = series.copy()
        series["timestamp"] = pd.to_datetime(series["timestamp"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
        series = series.dropna(subset=["timestamp"]).reset_index(drop=True)
        asof_probe = pd.merge_asof(
            merged[["timestamp"]],
            series.rename(columns={"timestamp": f"{col}_source_ts", "value": f"{col}_source_value"}),
            left_on="timestamp",
            right_on=f"{col}_source_ts",
            direction="backward",
        )
        leak_check += int((asof_probe[f"{col}_source_ts"] > merged["timestamp"]).sum())

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged, {"macro_rows": int(len(macro_df)), "asof_safe_violations": int(leak_check)}


def _neutral_band(vol: float, tx_cost_bps: float, vol_scale: float) -> float:
    tx = tx_cost_bps / 10_000.0
    safe_vol = max(0.0, float(vol))
    return max(tx, tx + (vol_scale * safe_vol))


def _return_to_class(next_return: float, band: float) -> int:
    if next_return > band:
        return 1
    if next_return < -band:
        return -1
    return 0


def build_market_frame(df: pd.DataFrame, tx_cost_bps: float, neutral_vol_scale: float) -> pd.DataFrame:
    out = engineer_features(df.copy(), is_forex=str(df["symbol"].iloc[0]).endswith("=X"))
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["next_return"] = out["close"].shift(-1) / out["close"] - 1.0
    out["volatility_rolling_20"] = (
        pd.to_numeric(out.get("close"), errors="coerce")
        .pct_change()
        .rolling(window=20, min_periods=5)
        .std()
        .fillna(0.0)
    )
    out["neutral_band"] = out["volatility_rolling_20"].apply(
        lambda v: _neutral_band(float(v), tx_cost_bps=tx_cost_bps, vol_scale=neutral_vol_scale)
    )
    out["label_int"] = [
        _return_to_class(float(ret), float(band)) if np.isfinite(ret) else np.nan
        for ret, band in zip(out["next_return"], out["neutral_band"])
    ]
    out["label"] = out["label_int"].map(INT_TO_CLASS)
    out["close_log_return"] = np.log(out["close"] / out["close"].shift(1)).replace([np.inf, -np.inf], np.nan)
    out["close_log_return"] = out["close_log_return"].fillna(0.0)
    return out


def _candidate_technical_features(df: pd.DataFrame) -> list[str]:
    excluded = {
        "symbol",
        "timestamp",
        "next_return",
        "label",
        "label_int",
        "neutral_band",
        "interval",
        "exchange",
        "source_type",
        "ingestion_timestamp_utc",
        "ingestion_timestamp_ist",
        "schema_version",
        "quality_status",
    }
    features: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        low = col.lower()
        if any(token in low for token in ("next_", "future", "target", "prediction")):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return sorted(features)


def train_technical_bundle(train_df: pd.DataFrame, feature_columns: list[str]) -> TechnicalModelBundle:
    usable = train_df.dropna(subset=["label_int", "next_return"]).copy()
    if usable.empty:
        raise ValueError("No usable rows available for technical training.")

    X = usable[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y_class = usable["label_int"].astype(int).to_numpy()
    y_reg = usable["next_return"].astype(float).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    class_prior = {-1: 1 / 3, 0: 1 / 3, 1: 1 / 3}
    unique, counts = np.unique(y_class, return_counts=True)
    total = max(1, int(counts.sum()))
    for klass, count in zip(unique, counts):
        class_prior[int(klass)] = float(count / total)

    cls_model: LogisticRegression | None = None
    if len(unique) >= 2:
        cls_model = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=42,
        )
        cls_model.fit(Xs, y_class)

    reg_model: Ridge | None = None
    if len(usable) >= 10:
        reg_model = Ridge(alpha=1.0, random_state=42)
        reg_model.fit(Xs, y_reg)

    return TechnicalModelBundle(
        scaler=scaler,
        cls_model=cls_model,
        reg_model=reg_model,
        feature_columns=feature_columns,
        class_prior=class_prior,
        fallback_mean_return=float(np.mean(y_reg)),
    )


def predict_technical(bundle: TechnicalModelBundle, row: pd.Series) -> dict[str, Any]:
    x = (
        row[bundle.feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
        .reshape(1, -1)
    )
    xs = bundle.scaler.transform(x)

    if bundle.cls_model is not None:
        probs_raw = bundle.cls_model.predict_proba(xs)[0]
        class_order = list(bundle.cls_model.classes_)
        prob_map = {int(klass): float(prob) for klass, prob in zip(class_order, probs_raw)}
        prob_up = prob_map.get(1, 0.0)
        prob_neutral = prob_map.get(0, 0.0)
        prob_down = prob_map.get(-1, 0.0)
    else:
        prob_up = bundle.class_prior.get(1, 1 / 3)
        prob_neutral = bundle.class_prior.get(0, 1 / 3)
        prob_down = bundle.class_prior.get(-1, 1 / 3)

    if bundle.reg_model is not None:
        predicted_return = float(bundle.reg_model.predict(xs)[0])
    else:
        predicted_return = float(bundle.fallback_mean_return)

    score = float(np.clip(prob_up - prob_down, -1.0, 1.0))
    confidence = float(np.clip(max(prob_up, prob_neutral, prob_down), 0.0, 1.0))
    volatility = float(max(0.0, row.get("volatility_rolling_20", 0.0)))
    var_95 = float(-1.65 * volatility)
    es_95 = float(-2.06 * volatility)

    return {
        "technical_predicted_return": predicted_return,
        "technical_score": score,
        "technical_confidence": confidence,
        "technical_prob_up": prob_up,
        "technical_prob_neutral": prob_neutral,
        "technical_prob_down": prob_down,
        "technical_volatility": volatility,
        "technical_var_95": var_95,
        "technical_es_95": es_95,
    }


def train_regime_bundle(train_df: pd.DataFrame) -> RegimeModelBundle:
    if len(train_df) < 25:
        return RegimeModelBundle(hmm=None, pearl=None, ood=None, fitted=False)

    hmm = HMMRegimeModel(n_components=4, random_state=42)
    pearl = PearlMetaModel(adaptation_window=min(60, max(20, len(train_df) // 2)))
    ood = OODDetector(
        warning_mahalanobis=3.0,
        alien_mahalanobis=5.0,
        warning_kl=1.0,
        alien_kl=6.0,
    )
    try:
        hmm.fit(train_df)
        pearl.fit(train_df)
        ood.fit(train_df)
    except Exception:
        return RegimeModelBundle(hmm=None, pearl=None, ood=None, fitted=False)
    return RegimeModelBundle(hmm=hmm, pearl=pearl, ood=ood, fitted=True)


def _heuristic_regime_from_row(row: pd.Series) -> tuple[str, float, float]:
    ret_z = float(row.get("close_log_return_zscore", 0.0))
    vol = float(row.get("rolling_vol_20", 0.0))
    macro = float(row.get("macro_regime_index", row.get("macro_directional_flag", 0.0)))
    if abs(ret_z) >= 3.0 or vol >= 0.03:
        return "Crisis", 0.75, 0.8
    if ret_z >= 0.8:
        return "Bull", 0.25, 0.7
    if ret_z <= -0.8:
        return "Bear", 0.25, 0.7
    if abs(macro) >= 0.5:
        return "RBI-Band transition", 0.4, 0.6
    return "Sideways", 0.2, 0.55


def predict_regime(bundle: RegimeModelBundle, history_plus_current: pd.DataFrame) -> dict[str, Any]:
    current_row = history_plus_current.iloc[-1]
    if not bundle.fitted or bundle.hmm is None or bundle.pearl is None or bundle.ood is None:
        label, transition_prob, conf = _heuristic_regime_from_row(current_row)
        return {
            "regime_label": label,
            "regime_score": REGIME_TO_SCORE[label],
            "regime_confidence": conf,
            "regime_transition_probability": transition_prob,
            "regime_ood_warning": False,
            "regime_ood_alien": False,
            "regime_risk_level": "reduced_risk" if REGIME_TO_PROTECTIVE[label] else "full_risk",
        }

    hmm_pred = bundle.hmm.predict(history_plus_current)
    pearl_pred = bundle.pearl.predict(history_plus_current)
    ood_result = bundle.ood.detect(history_plus_current.tail(min(len(history_plus_current), 120)))

    votes = {
        hmm_pred.regime_state.value: float(hmm_pred.confidence),
        pearl_pred.regime_state.value: float(pearl_pred.confidence),
    }
    if hmm_pred.regime_state.value == pearl_pred.regime_state.value:
        votes[hmm_pred.regime_state.value] = votes[hmm_pred.regime_state.value] + 0.2
    regime_label = max(votes, key=votes.get)
    if ood_result.is_alien:
        regime_label = "Alien"

    transition_prob = float(np.clip((0.6 * hmm_pred.transition_probability) + (0.4 * pearl_pred.transition_probability), 0.0, 1.0))
    confidence = float(np.clip((0.55 * hmm_pred.confidence) + (0.45 * pearl_pred.confidence), 0.0, 1.0))
    if ood_result.is_warning:
        confidence *= 0.9
    if ood_result.is_alien:
        confidence *= 0.8

    return {
        "regime_label": regime_label,
        "regime_score": REGIME_TO_SCORE.get(regime_label, 0.0),
        "regime_confidence": float(np.clip(confidence, 0.0, 1.0)),
        "regime_transition_probability": transition_prob,
        "regime_ood_warning": bool(ood_result.is_warning),
        "regime_ood_alien": bool(ood_result.is_alien),
        "regime_risk_level": str(ood_result.risk_level.value),
    }


def _symbol_aliases(symbol: str) -> list[str]:
    table = {
        "RELIANCE.NS": ["reliance", "ril"],
        "TATASTEEL.NS": ["tata steel", "tatasteel"],
        "USDINR=X": ["usdinr", "usd inr", "usd/inr", "rupee", "inr"],
    }
    return table.get(symbol, [symbol.lower().replace(".ns", "").replace("=x", "")])


def load_textual_artifacts(canonical_path: Path, sidecar_path: Path) -> pd.DataFrame:
    if not canonical_path.exists():
        return pd.DataFrame()
    canonical = pd.read_parquet(canonical_path).copy()
    if canonical.empty:
        return canonical

    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True, errors="coerce")
    canonical = canonical.dropna(subset=["timestamp"]).reset_index(drop=True)
    canonical["headline"] = canonical.get("headline", "").fillna("")
    canonical["content"] = canonical.get("content", "").fillna("")
    canonical["text"] = (canonical["headline"].astype(str) + " " + canonical["content"].astype(str)).str.strip()

    if sidecar_path.exists():
        sidecar = pd.read_parquet(sidecar_path).copy()
        join_cols = [col for col in ["source_id", "ingestion_timestamp_utc", "confidence", "ttl_seconds", "quality_flags"] if col in sidecar.columns]
        if "source_id" in sidecar.columns:
            canonical = canonical.merge(sidecar[join_cols], how="left", on="source_id")
    return canonical


def assign_text_symbol(text_df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if text_df.empty:
        return text_df
    out = text_df.copy()
    out["symbol"] = None
    combined = (out["headline"].astype(str) + " " + out["content"].astype(str) + " " + out.get("url", "").astype(str)).str.lower()
    for symbol in symbols:
        aliases = _symbol_aliases(symbol)
        mask = pd.Series(False, index=out.index)
        for alias in aliases:
            mask = mask | combined.str.contains(alias, regex=False)
        out.loc[mask & out["symbol"].isna(), "symbol"] = symbol
    return out


def score_text_documents(text_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if text_df.empty:
        return text_df
    set_seed(seed)
    agent = SentimentAgent.from_default_components()
    rows: list[dict[str, Any]] = []
    for record in text_df.itertuples(index=False):
        payload = {
            "source_id": str(getattr(record, "source_id", "unknown")),
            "headline": str(getattr(record, "headline", "") or ""),
            "content": str(getattr(record, "content", "") or ""),
            "normalized_content": str(getattr(record, "text", "") or ""),
            "language": str(getattr(record, "language", "en") or "en"),
        }
        prediction = agent.score_textual_payload(payload, lane="fast", as_of_utc=getattr(record, "timestamp"))
        rows.append(
            {
                "source_id": payload["source_id"],
                "timestamp": pd.Timestamp(getattr(record, "timestamp")).tz_convert("UTC"),
                "symbol": getattr(record, "symbol", None),
                "headline": payload["headline"],
                "content": payload["content"],
                "sentiment_label": prediction.label.value,
                "sentiment_score": float(prediction.score),
                "sentiment_confidence": float(prediction.confidence),
                "freshness_state": prediction.freshness_state.value,
                "quality_status": prediction.quality_status.value,
                "manipulation_risk_score": float(prediction.manipulation_risk_score),
            }
        )
    return pd.DataFrame(rows)


def aggregate_hourly_sentiment(
    doc_df: pd.DataFrame,
    market_frames: dict[str, pd.DataFrame],
    lookback_hours: int,
    stale_hours: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if doc_df.empty:
        for symbol, market_df in market_frames.items():
            for ts in market_df["timestamp"]:
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": ts,
                        "sentiment_score": 0.0,
                        "sentiment_confidence": 0.0,
                        "sentiment_source_count": 0,
                        "sentiment_freshness": "missing",
                        "sentiment_stale_flag": True,
                        "sentiment_missing": True,
                        "sentiment_dispersion": 0.0,
                        "sentiment_novelty": 0.0,
                    }
                )
        return pd.DataFrame(rows)

    global_docs = doc_df.copy()
    global_docs = global_docs.sort_values("timestamp").reset_index(drop=True)

    for symbol, market_df in market_frames.items():
        symbol_docs = global_docs[global_docs["symbol"] == symbol].copy()
        fallback_global = symbol_docs.empty
        active_docs = global_docs if fallback_global else symbol_docs

        for ts in market_df["timestamp"]:
            window_start = ts - timedelta(hours=lookback_hours)
            window = active_docs[(active_docs["timestamp"] <= ts) & (active_docs["timestamp"] >= window_start)]
            if window.empty:
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": ts,
                        "sentiment_score": 0.0,
                        "sentiment_confidence": 0.0,
                        "sentiment_source_count": 0,
                        "sentiment_freshness": "missing",
                        "sentiment_stale_flag": True,
                        "sentiment_missing": True,
                        "sentiment_dispersion": 0.0,
                        "sentiment_novelty": 0.0,
                    }
                )
                continue

            ages = (ts - window["timestamp"]).dt.total_seconds() / 3600.0
            fresh_weight = np.where(ages <= stale_hours, 1.0, 0.4)
            conf = window["sentiment_confidence"].to_numpy(dtype=float)
            score = window["sentiment_score"].to_numpy(dtype=float)
            weighted = np.maximum(1e-6, conf * fresh_weight)
            agg_score = float(np.clip(np.sum(score * weighted) / np.sum(weighted), -1.0, 1.0))
            agg_conf = float(np.clip(np.mean(conf * fresh_weight), 0.0, 1.0))
            min_age = float(np.min(ages))
            freshness = "fresh" if min_age <= stale_hours else "stale"
            stale_flag = freshness == "stale"
            uniqueness = float(window["headline"].astype(str).nunique() / max(1, len(window)))

            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "sentiment_score": agg_score,
                    "sentiment_confidence": agg_conf,
                    "sentiment_source_count": int(len(window)),
                    "sentiment_freshness": freshness,
                    "sentiment_stale_flag": bool(stale_flag),
                    "sentiment_missing": False,
                    "sentiment_dispersion": float(np.std(score) if len(score) > 1 else 0.0),
                    "sentiment_novelty": uniqueness,
                }
            )
    return pd.DataFrame(rows)


def _consensus_row_to_class(score: float, neutral_band: float) -> int:
    if score > neutral_band:
        return 1
    if score < -neutral_band:
        return -1
    return 0


def _consensus_neutral_band(row: pd.Series, tx_cost_bps: float, neutral_vol_scale: float) -> float:
    vol = float(max(0.0, row.get("volatility", row.get("technical_volatility", 0.0))))
    return _neutral_band(vol, tx_cost_bps=tx_cost_bps, vol_scale=neutral_vol_scale)


def run_weighted_consensus(base_df: pd.DataFrame, tx_cost_bps: float, neutral_vol_scale: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in base_df.itertuples(index=False):
        tech_w, regime_w, sent_w = 0.42, 0.35, 0.23
        sentiment_missing = bool(getattr(row, "sentiment_missing"))
        sentiment_stale = bool(getattr(row, "sentiment_stale_flag"))
        regime_ood_warning = bool(getattr(row, "regime_ood_warning"))
        regime_ood_alien = bool(getattr(row, "regime_ood_alien"))

        if sentiment_missing:
            sent_w *= 0.05
        elif sentiment_stale:
            sent_w *= 0.4

        if regime_ood_warning:
            regime_w *= 1.1
            tech_w *= 0.9
        if regime_ood_alien:
            regime_w *= 1.2
            tech_w *= 0.8
            sent_w *= 0.6

        total_w = max(1e-9, tech_w + regime_w + sent_w)
        tech_w /= total_w
        regime_w /= total_w
        sent_w /= total_w

        raw_score = (
            tech_w * float(getattr(row, "technical_score"))
            + regime_w * float(getattr(row, "regime_score"))
            + sent_w * float(getattr(row, "sentiment_score"))
        )
        disagreement_count = int(getattr(row, "agent_sign_disagreement_count"))
        dispersion = float(getattr(row, "agent_score_dispersion"))

        risk_mode = "normal"
        if regime_ood_alien or (disagreement_count >= 2 and dispersion >= 0.9):
            risk_mode = "protective"
        elif regime_ood_warning or disagreement_count >= 2 or dispersion >= 0.6:
            risk_mode = "reduced"

        if risk_mode == "protective":
            score = 0.0
        elif risk_mode == "reduced":
            score = raw_score * 0.5
        else:
            score = raw_score

        base_conf = (
            tech_w * float(getattr(row, "technical_confidence"))
            + regime_w * float(getattr(row, "regime_confidence"))
            + sent_w * float(getattr(row, "sentiment_confidence"))
        )
        penalty = 0.15 * float(regime_ood_warning) + 0.20 * float(regime_ood_alien) + 0.10 * float(dispersion >= 0.6)
        confidence = float(np.clip(base_conf - penalty, 0.0, 1.0))

        row_series = pd.Series(row._asdict())
        nband = _consensus_neutral_band(row_series, tx_cost_bps=tx_cost_bps, neutral_vol_scale=neutral_vol_scale)
        pred_class = _consensus_row_to_class(score, nband)

        rows.append(
            {
                "symbol": getattr(row, "symbol"),
                "timestamp": getattr(row, "timestamp"),
                "split": getattr(row, "split"),
                "model": "weighted_baseline",
                "score": float(np.clip(score, -1.0, 1.0)),
                "confidence": confidence,
                "pred_class_int": pred_class,
                "pred_class": INT_TO_CLASS[pred_class],
                "risk_mode": risk_mode,
                "w_technical": tech_w,
                "w_regime": regime_w,
                "w_sentiment": sent_w,
            }
        )
    return pd.DataFrame(rows)


def _transition_gate(df: pd.DataFrame, threshold: float = 0.35, gain: float = 4.0) -> np.ndarray:
    signal = (
        0.55 * df["volatility"].abs().to_numpy(dtype=float)
        + 0.20 * df["macro_differential"].abs().to_numpy(dtype=float)
        + 0.15 * df["rbi_signal"].abs().to_numpy(dtype=float)
        + 0.10 * df["sentiment_quantile"].to_numpy(dtype=float)
    )
    centered = signal - threshold
    lstar = 1.0 / (1.0 + np.exp(-(gain * centered)))
    estar = 1.0 - np.exp(-(signal**2))
    use_estar = df["volatility"].to_numpy(dtype=float) >= threshold
    return np.where(use_estar, estar, lstar)


def train_challenger(train_df: pd.DataFrame, feature_columns: list[str]) -> ChallengerModel:
    X = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = train_df["actual_return"].to_numpy(dtype=float)
    g = _transition_gate(train_df)
    switch = float(np.quantile(g, 0.5))
    low_mask = g <= switch
    high_mask = ~low_mask
    if low_mask.sum() < 8 or high_mask.sum() < 8:
        low_mask = np.ones(len(train_df), dtype=bool)
        high_mask = np.ones(len(train_df), dtype=bool)
        switch = 0.5

    low_model = BayesianRidge()
    high_model = BayesianRidge()
    low_model.fit(X[low_mask], y[low_mask])
    high_model.fit(X[high_mask], y[high_mask])

    pred_train = ((1 - g) * low_model.predict(X)) + (g * high_model.predict(X))
    residual_std = float(np.std(y - pred_train))
    return ChallengerModel(
        low_model=low_model,
        high_model=high_model,
        feature_columns=feature_columns,
        residual_std=max(1e-6, residual_std),
        transition_switch=switch,
    )


def run_challenger_consensus(
    model: ChallengerModel,
    base_df: pd.DataFrame,
    tx_cost_bps: float,
    neutral_vol_scale: float,
) -> pd.DataFrame:
    X = base_df[model.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    gate = _transition_gate(base_df)
    low_pred = model.low_model.predict(X)
    high_pred = model.high_model.predict(X)
    raw_score = ((1 - gate) * low_pred) + (gate * high_pred)

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(base_df.itertuples(index=False)):
        disagreement = abs(float(low_pred[idx]) - float(high_pred[idx]))
        confidence = float(np.clip(math.exp(-disagreement / (model.residual_std + 1e-6)), 0.0, 1.0))
        score = float(np.clip(raw_score[idx], -1.0, 1.0))

        sentiment_missing = bool(getattr(row, "sentiment_missing"))
        regime_ood_alien = bool(getattr(row, "regime_ood_alien"))
        if sentiment_missing:
            score *= 0.85
            confidence *= 0.9
        if regime_ood_alien:
            score = 0.0
            confidence *= 0.7

        row_series = pd.Series(row._asdict())
        nband = _consensus_neutral_band(row_series, tx_cost_bps=tx_cost_bps, neutral_vol_scale=neutral_vol_scale)
        pred_class = _consensus_row_to_class(score, nband)
        rows.append(
            {
                "symbol": getattr(row, "symbol"),
                "timestamp": getattr(row, "timestamp"),
                "split": getattr(row, "split"),
                "model": "challenger_lstar_estar_bayesian",
                "score": score,
                "confidence": confidence,
                "pred_class_int": pred_class,
                "pred_class": INT_TO_CLASS[pred_class],
                "risk_mode": "normal" if pred_class != 0 else "reduced",
                "w_technical": float("nan"),
                "w_regime": float("nan"),
                "w_sentiment": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _directional_metrics(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray | None = None) -> dict[str, Any]:
    labels = [-1, 0, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {}
    for i, label in enumerate(labels):
        per_class[INT_TO_CLASS[label]] = {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    metrics["per_class"] = per_class
    if confidence is not None and len(confidence) == len(y_true):
        correctness = (y_true == y_pred).astype(float)
        metrics["confidence_brier"] = float(np.mean((confidence - correctness) ** 2))
    return metrics


def _derive_regime_truth(df: pd.DataFrame) -> pd.Series:
    vol = df["volatility"].fillna(0.0)
    ret = df["actual_return"].fillna(0.0)
    crisis_cut = float(np.quantile(vol, 0.9)) if len(vol) else 0.03
    high_move_cut = float(np.quantile(np.abs(ret), 0.9)) if len(ret) else 0.01
    truth: list[str] = []
    for _, row in df.iterrows():
        r = float(row["actual_return"])
        v = float(row["volatility"])
        macro = float(row.get("macro_differential", 0.0))
        if v >= crisis_cut or abs(r) >= high_move_cut:
            truth.append("Crisis")
        elif r > 0.002:
            truth.append("Bull")
        elif r < -0.002:
            truth.append("Bear")
        elif abs(macro) >= 0.5:
            truth.append("RBI-Band transition")
        else:
            truth.append("Sideways")
    return pd.Series(truth, index=df.index)


def evaluate_base_agents(base_df: pd.DataFrame, doc_df: pd.DataFrame) -> dict[str, Any]:
    eval_df = base_df.copy()
    y_true = eval_df["actual_label_int"].astype(int).to_numpy()

    tech_pred = np.array([_consensus_row_to_class(float(v), float(b)) for v, b in zip(eval_df["technical_score"], eval_df["neutral_band"])], dtype=int)
    technical_metrics = _directional_metrics(y_true, tech_pred, eval_df["technical_confidence"].to_numpy(dtype=float))
    technical_metrics.update(
        {
            "regression_rmse": float(np.sqrt(np.mean((eval_df["technical_predicted_return"] - eval_df["actual_return"]) ** 2))),
            "regression_mae": float(np.mean(np.abs(eval_df["technical_predicted_return"] - eval_df["actual_return"]))),
            "time_slice_stability_std_accuracy": float(
                eval_df.assign(
                    tech_correct=(tech_pred == y_true).astype(float),
                    slice=eval_df["timestamp"].dt.date.astype(str),
                )
                .groupby("slice")["tech_correct"]
                .mean()
                .std(ddof=0)
                if len(eval_df) > 0
                else 0.0
            ),
        }
    )

    regime_truth = _derive_regime_truth(eval_df)
    regime_pred = eval_df["regime_label"].astype(str)
    regime_labels = sorted(set(regime_truth.unique()) | set(regime_pred.unique()))
    regime_conf = confusion_matrix(regime_truth, regime_pred, labels=regime_labels).tolist()
    regime_metrics = {
        "accuracy": float(accuracy_score(regime_truth, regime_pred)),
        "macro_f1": float(f1_score(regime_truth, regime_pred, labels=regime_labels, average="macro", zero_division=0)),
        "labels": regime_labels,
        "confusion_matrix": regime_conf,
        "transition_stability_flip_rate": float((regime_pred != regime_pred.shift(1)).fillna(False).mean()),
        "ood_warning_rate": float(eval_df["regime_ood_warning"].mean()),
        "ood_alien_rate": float(eval_df["regime_ood_alien"].mean()),
    }

    sentiment_metrics: dict[str, Any] = {
        "status": "insufficient_labeled_docs",
        "doc_count": int(len(doc_df)),
        "macro_f1": None,
        "class_balance": {},
        "timestamp_alignment_violations": 0,
    }
    if not doc_df.empty and "realized_label_int" in doc_df.columns:
        labeled = doc_df.dropna(subset=["realized_label_int"]).copy()
        if len(labeled) >= 8:
            y_s_true = labeled["realized_label_int"].astype(int).to_numpy()
            y_s_pred = labeled["sentiment_label_int"].astype(int).to_numpy()
            sentiment_metrics = _directional_metrics(
                y_s_true,
                y_s_pred,
                labeled["sentiment_confidence"].to_numpy(dtype=float),
            )
            sentiment_metrics.update(
                {
                    "status": "ok",
                    "doc_count": int(len(doc_df)),
                    "timestamp_alignment_violations": int((labeled["timestamp"] > labeled["aligned_market_timestamp"]).sum()),
                    "class_balance": labeled["realized_label_int"].astype(int).value_counts().sort_index().to_dict(),
                }
            )

    return {"technical": technical_metrics, "regime": regime_metrics, "sentiment": sentiment_metrics}


def evaluate_consensus(pred_df: pd.DataFrame, base_df: pd.DataFrame) -> dict[str, Any]:
    merged = pred_df.merge(
        base_df[
            [
                "symbol",
                "timestamp",
                "split",
                "actual_label_int",
                "actual_return",
                "volatility",
                "sentiment_stale_flag",
                "sentiment_missing",
                "agent_sign_disagreement_count",
                "agent_score_dispersion",
            ]
        ],
        on=["symbol", "timestamp", "split"],
        how="left",
    )
    if merged.empty:
        return {}

    y_true = merged["actual_label_int"].astype(int).to_numpy()
    y_pred = merged["pred_class_int"].astype(int).to_numpy()
    metrics = _directional_metrics(y_true, y_pred, merged["confidence"].to_numpy(dtype=float))
    high_vol_cut = float(np.quantile(merged["volatility"], 0.75))

    def _slice_accuracy(mask: pd.Series) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(accuracy_score(y_true[mask.to_numpy()], y_pred[mask.to_numpy()]))

    metrics["slice_accuracy"] = {
        "high_volatility": _slice_accuracy(merged["volatility"] >= high_vol_cut),
        "stale_or_missing_sentiment": _slice_accuracy(merged["sentiment_stale_flag"] | merged["sentiment_missing"]),
        "strong_disagreement": _slice_accuracy(
            (merged["agent_sign_disagreement_count"] >= 2) | (merged["agent_score_dispersion"] >= 0.6)
        ),
        "normal": _slice_accuracy(merged["volatility"] < high_vol_cut),
    }

    positions = merged["pred_class_int"].astype(float)
    shifted = positions.shift(1).fillna(0.0)
    costs = 0.0005 * (positions - shifted).abs()
    utility = (positions * merged["actual_return"]) - costs
    metrics["proxy_utility_after_costs"] = float(utility.sum())
    metrics["proxy_utility_mean_per_bar"] = float(utility.mean())
    metrics["robust_disagreement_neutral_rate"] = float(
        merged.loc[(merged["agent_sign_disagreement_count"] >= 2), "pred_class_int"].eq(0).mean()
        if int((merged["agent_sign_disagreement_count"] >= 2).sum()) > 0
        else 0.0
    )
    return metrics


def build_doc_level_labels(doc_df: pd.DataFrame, market_frames: dict[str, pd.DataFrame], tx_cost_bps: float, neutral_vol_scale: float) -> pd.DataFrame:
    if doc_df.empty:
        return doc_df

    benchmark_symbol = next(iter(market_frames.keys()))
    benchmark = market_frames[benchmark_symbol].sort_values("timestamp").reset_index(drop=True)
    benchmark["timestamp"] = pd.to_datetime(benchmark["timestamp"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    benchmark = benchmark[["timestamp", "next_return", "volatility_rolling_20"]].dropna()
    if benchmark.empty:
        return doc_df

    left = doc_df.sort_values("timestamp").copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    aligned = pd.merge_asof(
        left,
        benchmark.rename(columns={"timestamp": "aligned_market_timestamp"}),
        left_on="timestamp",
        right_on="aligned_market_timestamp",
        direction="forward",
    )
    aligned["aligned_market_timestamp"] = pd.to_datetime(aligned["aligned_market_timestamp"], utc=True, errors="coerce")
    aligned["realized_label_int"] = [
        _return_to_class(
            float(ret) if np.isfinite(ret) else 0.0,
            _neutral_band(float(vol), tx_cost_bps=tx_cost_bps, vol_scale=neutral_vol_scale),
        )
        if pd.notna(ts)
        else np.nan
        for ret, vol, ts in zip(
            aligned.get("next_return", pd.Series([np.nan] * len(aligned))),
            aligned.get("volatility_rolling_20", pd.Series([0.0] * len(aligned))),
            aligned["aligned_market_timestamp"],
        )
    ]
    aligned["sentiment_label_int"] = aligned["sentiment_label"].map(
        {"negative": -1, "neutral": 0, "positive": 1}
    )
    return aligned


def _consensus_training_features(df: pd.DataFrame) -> list[str]:
    return [
        "technical_score",
        "technical_confidence",
        "regime_score",
        "regime_confidence",
        "sentiment_score",
        "sentiment_confidence",
        "sentiment_stale_flag",
        "sentiment_missing",
        "volatility",
        "macro_differential",
        "rbi_signal",
        "sentiment_quantile",
        "crisis_probability",
        "agent_sign_disagreement_count",
        "agent_score_dispersion",
    ]


def select_consensus_training_rows(base_df: pd.DataFrame) -> pd.DataFrame:
    return base_df[base_df["split"] == "train_oos"].copy()


def run_walk_forward(
    symbol_frames: dict[str, pd.DataFrame],
    sentiment_hourly: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    model_artifacts: dict[str, Any] = {"technical": {}, "regime": {}}
    sentiment_lookup = (
        sentiment_hourly.set_index(["symbol", "timestamp"]).sort_index()
        if not sentiment_hourly.empty
        else pd.DataFrame()
    )

    for symbol, df in symbol_frames.items():
        frame = df.sort_values("timestamp").reset_index(drop=True).copy()
        usable_last_idx = len(frame) - 2
        if usable_last_idx <= config.min_train_rows:
            continue

        test_start = int(len(frame) * (1.0 - config.test_size_ratio))
        test_start = max(test_start, config.min_train_rows + 1)
        feature_columns = _candidate_technical_features(frame)
        if len(feature_columns) < 5:
            continue

        final_tech_model: TechnicalModelBundle | None = None
        final_regime_model: RegimeModelBundle | None = None

        for idx in range(config.min_train_rows, usable_last_idx + 1):
            train_df = frame.iloc[:idx].copy()
            current = frame.iloc[idx].copy()
            if train_df["label_int"].dropna().shape[0] < 25:
                continue

            try:
                tech_bundle = train_technical_bundle(train_df, feature_columns)
                tech_pred = predict_technical(tech_bundle, current)
                final_tech_model = tech_bundle
            except Exception:
                continue

            regime_bundle = train_regime_bundle(train_df)
            history_plus_current = pd.concat([train_df, frame.iloc[[idx]]], axis=0, ignore_index=True)
            regime_pred = predict_regime(regime_bundle, history_plus_current)
            final_regime_model = regime_bundle

            sentiment_row = {
                "sentiment_score": 0.0,
                "sentiment_confidence": 0.0,
                "sentiment_source_count": 0,
                "sentiment_freshness": "missing",
                "sentiment_stale_flag": True,
                "sentiment_missing": True,
                "sentiment_dispersion": 0.0,
                "sentiment_novelty": 0.0,
            }
            if isinstance(sentiment_lookup, pd.DataFrame) and not sentiment_lookup.empty:
                key = (symbol, current["timestamp"])
                if key in sentiment_lookup.index:
                    entry = sentiment_lookup.loc[key]
                    if isinstance(entry, pd.DataFrame):
                        entry = entry.iloc[-1]
                    sentiment_row = {
                        "sentiment_score": float(entry["sentiment_score"]),
                        "sentiment_confidence": float(entry["sentiment_confidence"]),
                        "sentiment_source_count": int(entry["sentiment_source_count"]),
                        "sentiment_freshness": str(entry["sentiment_freshness"]),
                        "sentiment_stale_flag": bool(entry["sentiment_stale_flag"]),
                        "sentiment_missing": bool(entry["sentiment_missing"]),
                        "sentiment_dispersion": float(entry["sentiment_dispersion"]),
                        "sentiment_novelty": float(entry["sentiment_novelty"]),
                    }

            score_signs = [
                np.sign(float(tech_pred["technical_score"])),
                np.sign(float(regime_pred["regime_score"])),
                np.sign(float(sentiment_row["sentiment_score"])),
            ]
            disagreement_count = int(len(set(score_signs)) - 1)
            score_dispersion = float(
                np.std(
                    [
                        float(tech_pred["technical_score"]),
                        float(regime_pred["regime_score"]),
                        float(sentiment_row["sentiment_score"]),
                    ]
                )
            )

            split = "test" if idx >= test_start else "train_oos"
            outputs.append(
                {
                    "symbol": symbol,
                    "timestamp": current["timestamp"],
                    "split": split,
                    "actual_return": float(current["next_return"]),
                    "actual_label_int": int(current["label_int"]),
                    "actual_label": str(current["label"]),
                    "neutral_band": float(current["neutral_band"]),
                    **tech_pred,
                    **regime_pred,
                    **sentiment_row,
                    "volatility": float(current.get("volatility_rolling_20", 0.0)),
                    "macro_differential": float(current.get("macro_regime_shock", 0.0)),
                    "rbi_signal": float(current.get("macro_directional_flag", 0.0)),
                    "sentiment_quantile": float(np.clip((sentiment_row["sentiment_score"] + 1.0) / 2.0, 0.0, 1.0)),
                    "crisis_probability": float(
                        np.clip(
                            0.5 * min(1.0, current.get("volatility_rolling_20", 0.0) / 0.03)
                            + 0.5 * RISK_LEVEL_TO_FLOAT.get(regime_pred["regime_risk_level"], 0.0),
                            0.0,
                            1.0,
                        )
                    ),
                    "agent_sign_disagreement_count": disagreement_count,
                    "agent_score_dispersion": score_dispersion,
                }
            )

        model_artifacts["technical"][symbol] = final_tech_model
        model_artifacts["regime"][symbol] = final_regime_model

    if not outputs:
        return pd.DataFrame(), model_artifacts
    base_df = pd.DataFrame(outputs).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return base_df, model_artifacts


def build_ablation_table(base_df: pd.DataFrame, weighted_pred: pd.DataFrame) -> pd.DataFrame:
    merged = weighted_pred.merge(base_df, on=["symbol", "timestamp", "split"], how="left", suffixes=("", "_base"))
    records: list[dict[str, Any]] = []

    def _evaluate_variant(name: str, scores: np.ndarray) -> dict[str, Any]:
        pred = np.array(
            [
                _consensus_row_to_class(float(score), float(band))
                for score, band in zip(scores, merged["neutral_band"].to_numpy(dtype=float))
            ],
            dtype=int,
        )
        y_true = merged["actual_label_int"].astype(int).to_numpy()
        return {
            "variant": name,
            "accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        }

    records.append(_evaluate_variant("weighted_consensus_baseline", merged["score"].to_numpy(dtype=float)))
    records.append(
        _evaluate_variant(
            "no_sentiment",
            0.6 * merged["technical_score"].to_numpy(dtype=float) + 0.4 * merged["regime_score"].to_numpy(dtype=float),
        )
    )
    records.append(
        _evaluate_variant(
            "no_regime",
            0.7 * merged["technical_score"].to_numpy(dtype=float) + 0.3 * merged["sentiment_score"].to_numpy(dtype=float),
        )
    )
    records.append(
        _evaluate_variant(
            "simple_average",
            (
                merged["technical_score"].to_numpy(dtype=float)
                + merged["regime_score"].to_numpy(dtype=float)
                + merged["sentiment_score"].to_numpy(dtype=float)
            )
            / 3.0,
        )
    )
    return pd.DataFrame(records)


def choose_recommendation(
    weighted_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    if not challenger_metrics:
        return {
            "recommended_model": "weighted_baseline",
            "reason": "Challenger model unavailable due insufficient training rows.",
        }
    w_acc = float(weighted_metrics.get("accuracy", 0.0))
    c_acc = float(challenger_metrics.get("accuracy", 0.0))
    w_hv = weighted_metrics.get("slice_accuracy", {}).get("high_volatility")
    c_hv = challenger_metrics.get("slice_accuracy", {}).get("high_volatility")

    if (
        (c_acc >= w_acc + 0.02)
        and (c_hv is None or w_hv is None or float(c_hv) >= float(w_hv) - 0.02)
    ):
        return {
            "recommended_model": "challenger_lstar_estar_bayesian",
            "reason": "Challenger shows clear out-of-sample uplift with no major instability on high-vol slices.",
        }
    return {
        "recommended_model": "weighted_baseline",
        "reason": "Baseline is more stable; challenger does not deliver clear robust out-of-sample uplift.",
    }


def build_markdown_report(
    config: PipelineConfig,
    run_dir: Path,
    data_overview: dict[str, Any],
    leakage_checks: dict[str, Any],
    base_metrics: dict[str, Any],
    weighted_test_metrics: dict[str, Any],
    challenger_test_metrics: dict[str, Any] | None,
    ablations: pd.DataFrame,
    recommendation: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Phase-2 Offline Training Report (Technical -> Regime -> Sentiment -> Consensus)")
    lines.append("")
    lines.append(f"- Generated at (UTC): {datetime.now(UTC).isoformat()}")
    lines.append(f"- Symbols: {', '.join(config.symbols)}")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append("")
    lines.append("## Data Used")
    lines.append(f"- Hourly OHLCV root: `{config.silver_ohlcv_root}`")
    lines.append(f"- Macro root: `{config.silver_macro_root}`")
    lines.append(f"- Textual canonical artifact: `{config.textual_canonical_path}`")
    lines.append(f"- Textual sidecar artifact: `{config.textual_sidecar_path}`")
    lines.append(f"- Data overview: `{json.dumps(data_overview)}`")
    lines.append("")
    lines.append("## Method")
    lines.append("- Chronological expanding-window walk-forward used for base agents.")
    lines.append("- Consensus trained only on out-of-sample base-agent predictions from pre-test windows.")
    lines.append("- Latest contiguous block reserved as untouched final test set.")
    lines.append("- Weighted baseline uses protective safeguards; challenger is Bayesian LSTAR/ESTAR-style.")
    lines.append("")
    lines.append("## Leakage Checks")
    lines.append(f"- Checks: `{json.dumps(leakage_checks)}`")
    lines.append("")
    lines.append("## Base Agent Metrics")
    lines.append(f"- Technical: `{json.dumps(base_metrics.get('technical', {}))}`")
    lines.append(f"- Regime: `{json.dumps(base_metrics.get('regime', {}))}`")
    lines.append(f"- Sentiment: `{json.dumps(base_metrics.get('sentiment', {}))}`")
    lines.append("")
    lines.append("## Consensus Metrics (Final Test)")
    lines.append(f"- Weighted baseline: `{json.dumps(weighted_test_metrics)}`")
    lines.append(f"- Challenger: `{json.dumps(challenger_test_metrics)}`")
    lines.append("")
    lines.append("## Ablations")
    if not ablations.empty:
        for row in ablations.itertuples(index=False):
            lines.append(f"- {row.variant}: accuracy={row.accuracy:.4f}, macro_f1={row.macro_f1:.4f}")
    else:
        lines.append("- No ablation rows available.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- Recommended now: **{recommendation['recommended_model']}**")
    lines.append(f"- Reason: {recommendation['reason']}")
    lines.append("")
    lines.append("## Assumptions and Limitations")
    lines.append("- Available local history is limited; long-horizon (2019+) validation is not currently possible from local files.")
    lines.append("- Textual data coverage is sparse and mostly point-in-time artifacts; sentiment is run with graceful degradation.")
    lines.append("- Results are offline research validation only; no live execution logic is touched.")
    return "\n".join(lines) + "\n"


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    set_seed(config.seed)
    run_id = datetime.now(UTC).strftime("phase2_consensus_%Y%m%d_%H%M%S")
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    symbol_frames: dict[str, pd.DataFrame] = {}
    data_overview: dict[str, Any] = {"symbols": {}, "macro_rows": 0, "doc_rows": 0}
    macro = _load_macro_frame(config.silver_macro_root)
    data_overview["macro_rows"] = int(len(macro))

    macro_leak_total = 0
    for symbol in config.symbols:
        raw = _load_symbol_ohlcv(symbol, config.silver_ohlcv_root)
        if raw.empty:
            continue
        merged, leak_stats = merge_macro_asof(raw, macro)
        macro_leak_total += int(leak_stats["asof_safe_violations"])
        prepared = build_market_frame(merged, tx_cost_bps=config.tx_cost_bps, neutral_vol_scale=config.neutral_vol_scale)
        symbol_frames[symbol] = prepared
        data_overview["symbols"][symbol] = {
            "rows": int(len(prepared)),
            "start_utc": str(prepared["timestamp"].min()),
            "end_utc": str(prepared["timestamp"].max()),
        }

    text_raw = load_textual_artifacts(config.textual_canonical_path, config.textual_sidecar_path)
    text_raw = assign_text_symbol(text_raw, list(symbol_frames.keys()))
    text_scored = score_text_documents(text_raw, seed=config.seed)
    text_scored = build_doc_level_labels(
        text_scored,
        symbol_frames,
        tx_cost_bps=config.tx_cost_bps,
        neutral_vol_scale=config.neutral_vol_scale,
    )
    data_overview["doc_rows"] = int(len(text_scored))

    sentiment_hourly = aggregate_hourly_sentiment(
        doc_df=text_scored,
        market_frames=symbol_frames,
        lookback_hours=config.sentiment_lookback_hours,
        stale_hours=config.sentiment_stale_hours,
    )

    base_df, model_artifacts = run_walk_forward(symbol_frames, sentiment_hourly, config)
    if base_df.empty:
        raise ValueError("No out-of-sample base predictions were generated. Check data availability and min_train_rows.")

    weighted_pred = run_weighted_consensus(
        base_df=base_df,
        tx_cost_bps=config.tx_cost_bps,
        neutral_vol_scale=config.neutral_vol_scale,
    )

    challenger_pred: pd.DataFrame | None = None
    challenger_model: ChallengerModel | None = None
    challenger_metrics_test: dict[str, Any] | None = None
    feature_cols = _consensus_training_features(base_df)
    train_oos = select_consensus_training_rows(base_df)
    if (not config.skip_challenger) and len(train_oos) >= 30:
        challenger_model = train_challenger(train_oos.assign(actual_return=train_oos["actual_return"]), feature_cols)
        challenger_pred = run_challenger_consensus(
            challenger_model,
            base_df,
            tx_cost_bps=config.tx_cost_bps,
            neutral_vol_scale=config.neutral_vol_scale,
        )

    base_metrics = evaluate_base_agents(base_df, text_scored)
    weighted_test_metrics = evaluate_consensus(
        weighted_pred[weighted_pred["split"] == "test"].copy(),
        base_df[base_df["split"] == "test"].copy(),
    )
    if challenger_pred is not None:
        challenger_metrics_test = evaluate_consensus(
            challenger_pred[challenger_pred["split"] == "test"].copy(),
            base_df[base_df["split"] == "test"].copy(),
        )

    ablations = build_ablation_table(
        base_df=base_df[base_df["split"] == "test"].copy(),
        weighted_pred=weighted_pred[weighted_pred["split"] == "test"].copy(),
    )
    recommendation = choose_recommendation(weighted_test_metrics, challenger_metrics_test)

    leakage_checks = {
        "macro_asof_safe_violations": int(macro_leak_total),
        "text_timestamp_alignment_violations": int(
            (text_scored["timestamp"] > text_scored["aligned_market_timestamp"]).sum()
            if "aligned_market_timestamp" in text_scored.columns
            else 0
        ),
        "consensus_train_rows_from_oos_only": bool(set(train_oos["split"].unique()) == {"train_oos"}),
        "chronological_order_pass": bool(base_df["timestamp"].is_monotonic_increasing),
        "future_label_join_violations": int((base_df["actual_return"].isna()).sum()),
    }

    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    base_df.to_csv(run_dir / "artifacts" / "base_agent_oos_predictions.csv", index=False)
    weighted_pred.to_csv(run_dir / "artifacts" / "consensus_weighted_predictions.csv", index=False)
    sentiment_hourly.to_csv(run_dir / "artifacts" / "sentiment_hourly_features.csv", index=False)
    text_scored.to_csv(run_dir / "artifacts" / "sentiment_document_predictions.csv", index=False)
    if challenger_pred is not None:
        challenger_pred.to_csv(run_dir / "artifacts" / "consensus_challenger_predictions.csv", index=False)
    ablations.to_csv(run_dir / "artifacts" / "consensus_ablations_test.csv", index=False)

    save_json(
        run_dir / "metrics.json",
        {
            "data_overview": data_overview,
            "leakage_checks": leakage_checks,
            "base_agents": base_metrics,
            "consensus_weighted_test": weighted_test_metrics,
            "consensus_challenger_test": challenger_metrics_test,
            "recommendation": recommendation,
        },
    )
    save_json(run_dir / "run_config.json", config.__dict__)

    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for symbol, model in model_artifacts["technical"].items():
        if model is not None:
            save_pickle(model_dir / f"technical_{sanitize_symbol(symbol)}.pkl", model)
    for symbol, model in model_artifacts["regime"].items():
        if model is not None:
            save_pickle(model_dir / f"regime_{sanitize_symbol(symbol)}.pkl", model)
    if challenger_model is not None:
        save_pickle(model_dir / "consensus_challenger.pkl", challenger_model)
    save_json(
        model_dir / "consensus_weighted_config.json",
        {"technical_base": 0.42, "regime_base": 0.35, "sentiment_base": 0.23},
    )

    report_text = build_markdown_report(
        config=config,
        run_dir=run_dir,
        data_overview=data_overview,
        leakage_checks=leakage_checks,
        base_metrics=base_metrics,
        weighted_test_metrics=weighted_test_metrics,
        challenger_test_metrics=challenger_metrics_test,
        ablations=ablations,
        recommendation=recommendation,
    )
    report_path = run_dir / "phase2_consensus_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    result = {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "metrics_path": str(run_dir / "metrics.json"),
        "recommendation": recommendation,
        "data_overview": data_overview,
    }
    save_json(run_dir / "run_manifest.json", result)
    return result
