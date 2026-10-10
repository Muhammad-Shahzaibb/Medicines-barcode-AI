from typing import Any, Optional

from pydantic import BaseModel, Field


class Medicine(BaseModel):
    gtin: Optional[str] = None
    batch_no: Optional[str] = None
    lot: Optional[str] = None
    mfg_date: Optional[str] = None  # ISO YYYY-MM-DD (month-only dates -> day 01)
    exp_date: Optional[str] = None
    serial_number: Optional[str] = None
    extraction_method: str = "paddleocr_vl"
    confidence: float = 0.0


class ExtractionResponse(BaseModel):
    source_image: str
    medicines: list[Medicine] = Field(default_factory=list)
    pipeline_steps: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    debug: Optional[dict[str, Any]] = None
