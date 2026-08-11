from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MedicineRecord:
    gtin: str | None = None
    batch_no: str | None = None
    lot: str | None = None
    mfg_date: str | None = None
    exp_date: str | None = None
    serial_number: str | None = None
    extraction_method: str = "unknown"
    confidence: float = 0.0
    raw_barcode: str | None = None
    raw_ocr: str | None = None
    source_fields: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gtin": self.gtin,
            "batch_no": self.batch_no,
            "lot": self.lot,
            "mfg_date": self.mfg_date,
            "exp_date": self.exp_date,
            "serial_number": self.serial_number,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
        }

    def filled_count(self) -> int:
        return sum(
            1
            for value in (
                self.gtin,
                self.batch_no,
                self.lot,
                self.mfg_date,
                self.exp_date,
                self.serial_number,
            )
            if value
        )

    def is_complete(self) -> bool:
        return self.gtin is not None and (self.batch_no or self.lot) and self.exp_date is not None

    def needs_mfg_enrichment(self) -> bool:
        return self.mfg_date is None


@dataclass
class ExtractionResult:
    source_image: str
    medicines: list[MedicineRecord] = field(default_factory=list)
    pipeline_steps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image": self.source_image,
            "medicines": [m.to_dict() for m in self.medicines],
            "pipeline_steps": self.pipeline_steps,
            "errors": self.errors,
        }
