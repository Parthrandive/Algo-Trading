import json
from typing import Optional
from datetime import datetime
import pandas as pd
from sqlalchemy import select, func, and_

from src.db.connection import get_engine
from src.db.models import (
    BacktestRunDB,
    ConsensusSignalDB,
    CorporateActionDB,
    MarketDataQualityDB,
    ModelCardDB,
    OHLCVBar,
    PredictionLogDB,
    RegimePredictionDB,
    SentimentScoreDB,
    TechnicalPredictionDB,
)

def get_latest_timestamp(symbol: str) -> Optional[datetime]:
    """
    Find the maximum timestamp for a symbol currently stored in the database.
    Useful for gap detection before fetching new data.
    Provides O(1) loop-up via the B-tree index, replacing the O(N) Parquet scan.
    """
    engine = get_engine()
    
    # Needs a session-less execute
    with engine.connect() as conn:
        stmt = select(func.max(OHLCVBar.timestamp)).where(OHLCVBar.symbol == symbol)
        result = conn.execute(stmt).scalar_one_or_none()
        return result

def get_bars(symbol: str, start: datetime, end: datetime, interval: str = "1h") -> pd.DataFrame:
    """
    Fetches bars for a specific symbol, interval, and time range.
    Returns a pandas DataFrame sorted by timestamp.
    """
    engine = get_engine()
    
    stmt = select(OHLCVBar).where(
        and_(
            OHLCVBar.symbol == symbol,
            OHLCVBar.interval == interval,
            OHLCVBar.timestamp >= start,
            OHLCVBar.timestamp <= end
        )
    ).order_by(OHLCVBar.timestamp.asc())
    
    # Read sql directly into pandas dataframe
    df = pd.read_sql(stmt, engine)
    
    # Ensure correct types
    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    return df

def get_corporate_actions(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetches corporate actions for a specific symbol within an ex_date range.
    """
    engine = get_engine()
    
    stmt = select(CorporateActionDB).where(
        and_(
            CorporateActionDB.symbol == symbol,
            CorporateActionDB.ex_date >= start,
            CorporateActionDB.ex_date <= end
        )
    ).order_by(CorporateActionDB.ex_date.asc())
    
    df = pd.read_sql(stmt, engine)
    if not df.empty and 'ex_date' in df.columns:
        df['ex_date'] = pd.to_datetime(df['ex_date'])
        
    return df


def get_market_data_quality(symbol: str, interval: str, dataset_type: str = "historical") -> dict | None:
    engine = get_engine()
    stmt = select(MarketDataQualityDB).where(
        and_(
            MarketDataQualityDB.symbol == symbol,
            MarketDataQualityDB.interval == interval,
            MarketDataQualityDB.dataset_type == dataset_type,
        )
    )
    df = pd.read_sql(stmt, engine)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    if row.get("details_json"):
        row["details_json"] = json.loads(row["details_json"])
    for column in ("first_timestamp", "last_timestamp", "updated_at"):
        if row.get(column) is not None:
            row[column] = pd.to_datetime(row[column], utc=True, errors="coerce")
    return row


def is_symbol_train_ready(symbol: str, interval: str, dataset_type: str = "historical") -> bool:
    quality = get_market_data_quality(symbol, interval, dataset_type=dataset_type)
    if quality is None:
        return False
    return bool(quality.get("train_ready"))


def _read_dataframe(stmt) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(stmt, engine)
    for column in ("timestamp", "run_timestamp", "backtest_start", "backtest_end", "created_at", "updated_at"):
        if column in df.columns and not df.empty:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    return df


def _parse_json_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns and not df.empty:
        df[column] = df[column].apply(lambda value: None if value in (None, "") else json.loads(value))
    return df


def get_technical_predictions(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    stmt = select(TechnicalPredictionDB).where(
        and_(
            TechnicalPredictionDB.symbol == symbol,
            TechnicalPredictionDB.timestamp >= start,
            TechnicalPredictionDB.timestamp <= end,
        )
    ).order_by(TechnicalPredictionDB.timestamp.asc())
    return _read_dataframe(stmt)


def get_regime_predictions(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    stmt = select(RegimePredictionDB).where(
        and_(
            RegimePredictionDB.symbol == symbol,
            RegimePredictionDB.timestamp >= start,
            RegimePredictionDB.timestamp <= end,
        )
    ).order_by(RegimePredictionDB.timestamp.asc())
    return _parse_json_column(_read_dataframe(stmt), "details_json")


def get_sentiment_scores(
    symbol: str | None,
    start: datetime,
    end: datetime,
    lane: str | None = None,
) -> pd.DataFrame:
    filters = [
        SentimentScoreDB.timestamp >= start,
        SentimentScoreDB.timestamp <= end,
    ]
    if symbol is None:
        filters.append(SentimentScoreDB.symbol.is_(None))
    else:
        filters.append(SentimentScoreDB.symbol == symbol)
    if lane is not None:
        filters.append(SentimentScoreDB.lane == lane)

    stmt = select(SentimentScoreDB).where(and_(*filters)).order_by(SentimentScoreDB.timestamp.asc())
    return _read_dataframe(stmt)


def get_consensus_signals(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    stmt = select(ConsensusSignalDB).where(
        and_(
            ConsensusSignalDB.symbol == symbol,
            ConsensusSignalDB.timestamp >= start,
            ConsensusSignalDB.timestamp <= end,
        )
    ).order_by(ConsensusSignalDB.timestamp.asc())
    return _read_dataframe(stmt)


def get_prediction_log(agent: str, start: datetime, end: datetime) -> pd.DataFrame:
    stmt = select(PredictionLogDB).where(
        and_(
            PredictionLogDB.agent == agent,
            PredictionLogDB.timestamp >= start,
            PredictionLogDB.timestamp <= end,
        )
    ).order_by(PredictionLogDB.timestamp.asc(), PredictionLogDB.id.asc())
    return _parse_json_column(_read_dataframe(stmt), "prediction_json")


def get_model_card(model_id: str) -> dict | None:
    engine = get_engine()
    stmt = select(ModelCardDB).where(ModelCardDB.model_id == model_id)
    df = pd.read_sql(stmt, engine)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    row["metadata_json"] = json.loads(row["metadata_json"]) if row.get("metadata_json") else None
    row["performance_json"] = json.loads(row["performance_json"]) if row.get("performance_json") else None
    for column in ("created_at", "updated_at"):
        if row.get(column) is not None:
            row[column] = pd.to_datetime(row[column], utc=True, errors="coerce")
    return row


def get_backtest_runs(model_id: str) -> pd.DataFrame:
    stmt = select(BacktestRunDB).where(BacktestRunDB.model_id == model_id).order_by(
        BacktestRunDB.run_timestamp.desc(),
        BacktestRunDB.id.desc(),
    )
    return _parse_json_column(_read_dataframe(stmt), "params_json")
