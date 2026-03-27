from datetime import UTC, datetime, timedelta

import pytest

import scripts.run_phase3_teacher_dry_run as dry_run


def test_parser_defaults_to_auto_window_mode() -> None:
    parser = dry_run.build_parser()
    args = parser.parse_args(["--symbols", "RELIANCE.NS,TCS.NS"])

    assert args.start is None
    assert args.end is None
    assert args.auto_window_days == 7


def test_resolve_window_uses_explicit_bounds() -> None:
    start, end, mode = dry_run._resolve_window(
        start_raw="2026-03-01T00:00:00Z",
        end_raw="2026-03-03T00:00:00Z",
        symbols=["RELIANCE.NS"],
        database_url=None,
        auto_window_days=7,
    )

    assert mode == "explicit"
    assert start == datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    assert end == datetime(2026, 3, 3, 0, 0, tzinfo=UTC)


def test_resolve_window_auto_mode_uses_latest_phase2_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    latest = datetime(2026, 3, 27, 16, 45, tzinfo=UTC)

    monkeypatch.setattr(dry_run, "_resolve_latest_phase2_timestamp", lambda symbols, database_url: latest)

    start, end, mode = dry_run._resolve_window(
        start_raw=None,
        end_raw=None,
        symbols=["RELIANCE.NS", "TCS.NS"],
        database_url="sqlite:///tmp.db",
        auto_window_days=5,
    )

    assert mode == "auto_latest_phase2"
    assert end == latest
    assert start == latest - timedelta(days=5)


def test_resolve_window_rejects_partial_explicit_bounds() -> None:
    with pytest.raises(ValueError, match="Provide both --start and --end"):
        dry_run._resolve_window(
            start_raw="2026-03-01T00:00:00Z",
            end_raw=None,
            symbols=["RELIANCE.NS"],
            database_url=None,
            auto_window_days=7,
        )


def test_resolve_window_rejects_non_positive_auto_window_days() -> None:
    with pytest.raises(ValueError, match="auto-window-days must be >= 1"):
        dry_run._resolve_window(
            start_raw=None,
            end_raw=None,
            symbols=["RELIANCE.NS"],
            database_url=None,
            auto_window_days=0,
        )


def test_resolve_window_rejects_reversed_explicit_bounds() -> None:
    with pytest.raises(ValueError, match="end must be greater than or equal to start"):
        dry_run._resolve_window(
            start_raw="2026-03-03T00:00:00Z",
            end_raw="2026-03-01T00:00:00Z",
            symbols=["RELIANCE.NS"],
            database_url=None,
            auto_window_days=7,
        )
