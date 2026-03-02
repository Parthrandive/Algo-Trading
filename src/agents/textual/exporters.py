from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel

from src.schemas.text_sidecar import TextSidecarMetadata


@dataclass(frozen=True)
class TextualExportBatch:
    canonical_records: list[BaseModel]
    sidecar_records: list[TextSidecarMetadata]


class TextualExporter:
    def build_batch(
        self,
        canonical_records: Sequence[BaseModel],
        sidecar_records: Sequence[TextSidecarMetadata],
    ) -> TextualExportBatch:
        return TextualExportBatch(
            canonical_records=list(canonical_records),
            sidecar_records=list(sidecar_records),
        )

    def as_dict(self, batch: TextualExportBatch) -> dict[str, list[dict[str, object]]]:
        return {
            "canonical_records": [record.model_dump(mode="json") for record in batch.canonical_records],
            "sidecar_records": [record.model_dump(mode="json") for record in batch.sidecar_records],
        }
