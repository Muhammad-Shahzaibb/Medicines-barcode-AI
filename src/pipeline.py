from __future__ import annotations

from collections import Counter

from src.llm_extractor import extract_with_vision_llm
from src.models import ExtractionResult, MedicineRecord


def _harmonize_multi_unit(records: list[MedicineRecord]) -> list[MedicineRecord]:
    if len(records) <= 1:
        return records

    for field in ("gtin", "batch_no", "lot", "exp_date"):
        values = [getattr(r, field) for r in records if getattr(r, field)]
        if not values:
            continue
        most_common = Counter(values).most_common(1)[0][0]
        for record in records:
            setattr(record, field, most_common)

    return records


def process_image(
    image_bytes: bytes,
    source_name: str = "uploaded_image",
    mime: str = "image/jpeg",
) -> ExtractionResult:
    result = ExtractionResult(source_image=source_name)

    try:
        records = extract_with_vision_llm(image_bytes, mime=mime)
        result.pipeline_steps.append(f"vision_llm: extracted {len(records)} medicine(s)")
        result.medicines = _harmonize_multi_unit(records)
    except Exception as exc:
        result.errors.append(f"Vision extraction failed: {exc}")
        result.pipeline_steps.append("vision_llm: failed")

    if not result.medicines:
        result.errors.append("No medicine data could be extracted from this image.")

    return result
