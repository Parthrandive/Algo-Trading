import pytest
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo
from pydantic import ValidationError
from src.schemas.market_data import (
    Bar,
    CorporateAction,
    CorporateActionType,
    SourceType as MarketSourceType,
    Tick,
)
from src.schemas.macro_data import MacroIndicator, MacroIndicatorType, SourceType as MacroSourceType
from src.schemas.text_data import NewsArticle, SocialPost, EarningsTranscript, Language, SourceType as TextSourceType
from src.utils.schema_registry import SchemaRegistry

def test_tick_validation():
    # Valid Tick
    tick = Tick(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        source=MarketSourceType.OFFICIAL_API,
        price=2500.0,
        volume=100
    )
    assert tick.price == 2500.0

    # Invalid Price (Negative)
    with pytest.raises(ValidationError):
        Tick(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            source_type=MarketSourceType.OFFICIAL_API,
            price=-10.0,
            volume=100
        )

def test_bar_validation():
    # Valid Bar
    bar = Bar(
        symbol="TCS",
        timestamp=datetime.now(),
        source_type=MarketSourceType.BROKER_API,
        interval="1m",
        open=3000,
        high=3050,
        low=2990,
        close=3020,
        volume=500
    )
    assert bar.high >= bar.low

    # Invalid High < Low
    with pytest.raises(ValidationError):
        Bar(
            symbol="TCS",
            timestamp=datetime.now(),
            source_type=MarketSourceType.BROKER_API,
            interval="1m",
            open=3000,
            high=2900, # Invalid
            low=2990,
            close=3020,
            volume=500
        )

def test_bar_rejects_close_outside_high_low_bounds():
    with pytest.raises(ValidationError):
        Bar(
            symbol="TCS",
            timestamp=datetime.now(),
            source_type=MarketSourceType.BROKER_API,
            interval="1m",
            open=3000,
            high=3050,
            low=2990,
            close=3060,  # Invalid: close > high
            volume=500
        )

def test_corporate_action_validation_for_dividend_and_dates():
    with pytest.raises(ValidationError):
        CorporateAction(
            symbol="RELIANCE",
            timestamp=datetime.now(UTC),
            source_type=MarketSourceType.OFFICIAL_API,
            action_type=CorporateActionType.DIVIDEND,
            ex_date=datetime.now(UTC),
        )

    ex_date = datetime(2026, 2, 19, 0, 0, tzinfo=UTC)
    with pytest.raises(ValidationError):
        CorporateAction(
            symbol="RELIANCE",
            timestamp=datetime.now(UTC),
            source_type=MarketSourceType.OFFICIAL_API,
            action_type=CorporateActionType.DIVIDEND,
            value=10.0,
            ex_date=ex_date,
            record_date=ex_date - timedelta(days=1),
        )


def test_corporate_action_validation_for_ratio_types():
    with pytest.raises(ValidationError):
        CorporateAction(
            symbol="INFY",
            timestamp=datetime.now(UTC),
            source_type=MarketSourceType.BROKER_API,
            action_type=CorporateActionType.SPLIT,
            ex_date=datetime.now(UTC),
        )

    action = CorporateAction(
        symbol="INFY",
        timestamp=datetime.now(UTC),
        source_type=MarketSourceType.BROKER_API,
        action_type=CorporateActionType.RIGHTS,
        ratio="3/10",
        ex_date=datetime.now(UTC),
        record_date=datetime.now(UTC) + timedelta(days=1),
    )
    assert action.ratio == "3:10"

    with pytest.raises(ValidationError):
        Bar(
            symbol="TCS",
            timestamp=datetime.now(),
            source_type=MarketSourceType.BROKER_API,
            interval="1m",
            open=3000,
            high=3050,
            low=2990,
            close=2980,  # Invalid: close < low
            volume=500
        )

def test_macro_indicator():
    macro = MacroIndicator(
        indicator_name=MacroIndicatorType.CPI,
        value=5.6,
        unit="%",
        period="Monthly",
        timestamp=datetime.now(),
        source_type=MacroSourceType.OFFICIAL_API
    )
    assert macro.value == 5.6

def test_text_data():
    news = NewsArticle(
        source_id="news_123",
        timestamp=datetime.now(),
        content="Market hit all time high",
        source_type=TextSourceType.RSS_FEED,
        headline="Sensex crosses 80k",
        publisher="MoneyControl"
    )
    assert news.language == Language.EN
    
    transcript = EarningsTranscript(
         source_id="txn_456",
         timestamp=datetime.now(),
         content="CEO said growth is strong",
         source_type=TextSourceType.OFFICIAL_API,
         symbol="INFY",
         quarter="Q4",
         year=2025
    )
    assert transcript.symbol == "INFY"

def test_social_post_rejects_negative_engagement():
    with pytest.raises(ValidationError):
        SocialPost(
            source_id="social_123",
            timestamp=datetime.now(),
            content="Bullish thread",
            source_type=TextSourceType.SOCIAL_MEDIA,
            platform="X",
            likes=-1,
            shares=0
        )

    with pytest.raises(ValidationError):
        SocialPost(
            source_id="social_124",
            timestamp=datetime.now(),
            content="Bearish thread",
            source_type=TextSourceType.SOCIAL_MEDIA,
            platform="Reddit",
            likes=0,
            shares=-2
        )

def test_ingestion_timestamp_defaults_to_timezone_aware_utc():
    tick = Tick(
        symbol="SBIN",
        timestamp=datetime.now(),
        source_type=MarketSourceType.OFFICIAL_API,
        price=810.0,
        volume=1000
    )
    assert tick.ingestion_timestamp_utc.tzinfo is not None
    assert tick.ingestion_timestamp_ist.tzinfo is not None
    assert tick.ingestion_timestamp_utc.utcoffset() == datetime.now(UTC).utcoffset()

def test_provenance_alias_backward_compatibility():
    tick = Tick(
        symbol="ITC",
        timestamp=datetime.now(),
        source="official_api",
        price=450.0,
        volume=300,
        quality_flag="warn",
    )
    assert tick.source_type == MarketSourceType.OFFICIAL_API
    assert tick.source == MarketSourceType.OFFICIAL_API
    assert tick.quality_status.value == "warn"
    assert tick.quality_flag.value == "warn"

def test_rejects_inconsistent_utc_ist_ingestion_pair():
    utc_ts = datetime.now(UTC)
    inconsistent_ist = (utc_ts + timedelta(minutes=2)).astimezone(ZoneInfo("Asia/Kolkata"))

    with pytest.raises(ValidationError):
        Tick(
            symbol="ITC",
            timestamp=datetime.now(),
            source_type=MarketSourceType.OFFICIAL_API,
            price=451.0,
            volume=100,
            ingestion_timestamp_utc=utc_ts,
            ingestion_timestamp_ist=inconsistent_ist,
        )

def test_schema_registry():
    # Valid Retrieval
    model = SchemaRegistry.get_model("Tick_v1.0")
    assert model == Tick

    # Invalid Retrieval
    with pytest.raises(ValueError):
        SchemaRegistry.get_model("Unknown_v1.0")

    # Validate Data via Registry
    valid_data = {
        "symbol": "HDFCBANK",
        "timestamp": datetime.now(),
        "source_type": "official_api",
        "price": 1600.0,
        "volume": 5000
    }
    
    instance = SchemaRegistry.validate("Tick_v1.0", valid_data)
    assert isinstance(instance, Tick)
    assert instance.symbol == "HDFCBANK"
    
    # Extra fields rejected
    invalid_data = valid_data.copy()
    invalid_data["random_field"] = "should_fail"
    
    with pytest.raises(ValidationError):
        SchemaRegistry.validate("Tick_v1.0", invalid_data)
