
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from src.agents.sentinel.failover_client import DegradationState, FailoverSentinelClient
from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import CorporateAction, CorporateActionType, Tick, SourceType, QualityFlag

# Mock Tick for return values
MOCK_TICK = Tick(
    symbol="RELIANCE", 
    timestamp=datetime.now(timezone.utc), 
    source_type=SourceType.OFFICIAL_API, 
    price=100.0, 
    volume=100
)

MOCK_ACTION = CorporateAction(
    symbol="RELIANCE",
    timestamp=datetime.now(timezone.utc),
    source_type=SourceType.OFFICIAL_API,
    action_type=CorporateActionType.DIVIDEND,
    value=15.0,
    ex_date=datetime.now(timezone.utc),
    record_date=datetime.now(timezone.utc),
)

@pytest.fixture
def mock_primary():
    client = MagicMock(spec=NSEClientInterface)
    return client

@pytest.fixture
def mock_fallback():
    client = MagicMock(spec=NSEClientInterface)
    return client

def test_primary_success(mock_primary, mock_fallback):
    client = FailoverSentinelClient(mock_primary, [mock_fallback])
    mock_primary.get_stock_quote.return_value = MOCK_TICK
    
    result = client.get_stock_quote("RELIANCE")
    
    assert result == MOCK_TICK
    mock_primary.get_stock_quote.assert_called_once()
    mock_fallback.get_stock_quote.assert_not_called()
    assert client.degradation_state == DegradationState.NORMAL

def test_primary_failure_fallback_success(mock_primary, mock_fallback):
    client = FailoverSentinelClient(mock_primary, [mock_fallback], failure_threshold=3)
    mock_primary.get_stock_quote.side_effect = Exception("Primary failed")
    mock_fallback.get_stock_quote.return_value = MOCK_TICK
    
    result = client.get_stock_quote("RELIANCE")
    
    assert result.source_type == SourceType.FALLBACK_SCRAPER
    assert result.quality_status == QualityFlag.WARN
    mock_primary.get_stock_quote.assert_called_once()
    mock_fallback.get_stock_quote.assert_called_once()
    
    # Internal state checks
    assert client._primary_failures == 1
    assert client._is_primary_healthy is True  # Threshold 3, so still healthy
    assert client.degradation_state == DegradationState.REDUCE_ONLY

def test_primary_unhealthy_switch(mock_primary, mock_fallback):
    # Set threshold to 1 for immediate unhealthy state
    client = FailoverSentinelClient(mock_primary, [mock_fallback], failure_threshold=1)
    mock_primary.get_stock_quote.side_effect = Exception("Primary failed")
    mock_fallback.get_stock_quote.return_value = MOCK_TICK
    
    # First call - fails and marks unhealthy
    client.get_stock_quote("RELIANCE")
    assert client._is_primary_healthy is False
    assert client.degradation_state == DegradationState.REDUCE_ONLY
    
    # Second call - should skip primary
    mock_primary.reset_mock()
    mock_fallback.reset_mock()
    mock_fallback.get_stock_quote.return_value = MOCK_TICK
    
    client.get_stock_quote("RELIANCE")
    
    mock_primary.get_stock_quote.assert_not_called()
    mock_fallback.get_stock_quote.assert_called_once()
    assert client.degradation_state == DegradationState.REDUCE_ONLY

def test_recovery_logic(mock_primary, mock_fallback):
    # Fail threshold 1. Cooldown small but we will mock time
    client = FailoverSentinelClient(
        mock_primary,
        [mock_fallback],
        failure_threshold=1,
        cooldown_seconds=60,
        recovery_success_threshold=1,
    )
    
    # 1. Fail primary -> Unhealthy
    mock_primary.get_stock_quote.side_effect = Exception("Primary failed")
    mock_fallback.get_stock_quote.return_value = MOCK_TICK
    client.get_stock_quote("RELIANCE")
    assert client._is_primary_healthy is False
    assert client.degradation_state == DegradationState.REDUCE_ONLY
    
    # 2. Mock time forward past cooldown
    with patch('src.agents.sentinel.failover_client.time.time') as mock_time:
        # Initial failure time was likely whatever time.time() was. 
        # But wait, failover_client uses time.time().
        # Let's just manually set _last_failure_time to 0 and mock time.time() to 100
        client._last_failure_time = 0
        mock_time.return_value = 100 # > 60s cooldown
        
        # Reset primary to succeed
        mock_primary.reset_mock()
        mock_primary.get_stock_quote.side_effect = None
        mock_primary.get_stock_quote.return_value = MOCK_TICK
        
        # 3. Call should probe primary
        client.get_stock_quote("RELIANCE")
        
        # Should have called primary
        mock_primary.get_stock_quote.assert_called_once()
        # Should be healthy and recovered to normal
        assert client._is_primary_healthy is True
        assert client.degradation_state == DegradationState.NORMAL
        # Failures should be reset (or set to threshold - 1)
        assert client._primary_failures == 0 # Logic is max(0, 1-1) = 0 for threshold 1

def test_all_failure(mock_primary, mock_fallback):
    client = FailoverSentinelClient(mock_primary, [mock_fallback])
    mock_primary.get_stock_quote.side_effect = Exception("Primary failed")
    mock_fallback.get_stock_quote.side_effect = Exception("Fallback failed")
    
    with pytest.raises(RuntimeError, match="All clients failed"):
        client.get_stock_quote("RELIANCE")
    assert client.degradation_state == DegradationState.CLOSE_ONLY_ADVISORY


def test_corporate_actions_are_tagged_on_fallback(mock_primary, mock_fallback):
    client = FailoverSentinelClient(mock_primary, [mock_fallback], failure_threshold=1)
    mock_primary.get_corporate_actions.side_effect = Exception("Primary failed")
    mock_fallback.get_corporate_actions.return_value = [MOCK_ACTION]

    result = client.get_corporate_actions(
        symbol="RELIANCE",
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )

    assert len(result) == 1
    assert result[0].source_type == SourceType.FALLBACK_SCRAPER
    assert result[0].quality_status == QualityFlag.WARN
    assert client.degradation_state == DegradationState.REDUCE_ONLY


def test_corporate_actions_primary_empty_falls_back(mock_primary, mock_fallback):
    client = FailoverSentinelClient(mock_primary, [mock_fallback], failure_threshold=1)
    mock_primary.get_corporate_actions.return_value = []
    mock_fallback.get_corporate_actions.return_value = [MOCK_ACTION]

    result = client.get_corporate_actions(
        symbol="RELIANCE",
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )

    assert len(result) == 1
    assert mock_primary.get_corporate_actions.called
    assert mock_fallback.get_corporate_actions.called
