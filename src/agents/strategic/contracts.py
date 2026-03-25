from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.agents.strategic.config import STRATEGIC_EXEC_CONTRACT_VERSION
from src.agents.strategic.schemas import StrategicObservation, StrategicToExecutiveContract


class StrategicContractEnvelope(BaseModel):
    """
    Canonical handoff payload from strategic workers to strategic_executive.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: str = STRATEGIC_EXEC_CONTRACT_VERSION
    observation: StrategicObservation
    decision: StrategicToExecutiveContract
