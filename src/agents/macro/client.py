"""
MacroClientInterface — Protocol definition for all macro data source clients.

Every concrete client (MOSPIClient, RBIClient, etc.) must satisfy this Protocol.
The Protocol uses duck typing so concrete classes need NOT inherit from it;
they just need to implement `get_indicator`.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple, Protocol, Sequence

from src.schemas.macro_data import MacroIndicator, MacroIndicatorType


class DateRange(NamedTuple):
    """Inclusive date range for indicator queries."""

    start: date
    end: date


class MacroClientInterface(Protocol):
    """
    Protocol that every macro source client must satisfy.

    A client is responsible for fetching raw data from ONE logical source
    (e.g. MOSPI, RBI, NSDL/NSE) and returning fully-constructed
    ``MacroIndicator`` records — complete with provenance tags
    (source_type, schema_version, quality_status, ingestion timestamps).

    Implementations must be stateless with respect to date ranges;
    callers supply the range on every call.
    """

    def get_indicator(
        self,
        name: MacroIndicatorType,
        date_range: DateRange,
    ) -> Sequence[MacroIndicator]:
        """
        Fetch indicator values for the given date range.

        Parameters
        ----------
        name:
            The ``MacroIndicatorType`` enum member to fetch.
        date_range:
            Inclusive ``[start, end]`` date window.

        Returns
        -------
        A (possibly empty) sequence of ``MacroIndicator`` records.
        Records must carry full provenance: source_type, schema_version
        ``"1.1"``, quality_status, ingestion_timestamp_utc/ist.

        Raises
        ------
        NotImplementedError:
            If the client does not support the requested indicator.
        RuntimeError:
            If upstream fetch fails after exhausting retries.
        """
        ...

    @property
    def supported_indicators(self) -> frozenset[MacroIndicatorType]:
        """Return the set of indicators this client can serve."""
        ...
