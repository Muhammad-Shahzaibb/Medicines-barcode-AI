from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from src.gs1_parser import is_gs1_payload, parse_gs1
from src.models import MedicineRecord
from src.normalizer import normalize_gtin

try:
    import zxingcpp

    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False


def _decode_with_zxing(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    if not HAS_ZXING:
        return []
    results = zxingcpp.read_barcodes(image_bgr)
    decoded: list[dict[str, Any]] = []
    for result in results:
        if not result.valid:
            continue
        decoded.append(
            {
                "text": result.text,
                "format": str(result.format),
            }
        )
    return decoded


def _preprocess_variants(image_bgr: np.ndarray) -> list[np.ndarray]:
    variants = [image_bgr]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    for angle in (90, 180, 270):
        matrix = cv2.getRotationMatrix2D(
            (image_bgr.shape[1] / 2, image_bgr.shape[0] / 2), angle, 1.0
        )
        rotated = cv2.warpAffine(image_bgr, matrix, (image_bgr.shape[1], image_bgr.shape[0]))
        variants.append(rotated)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    return variants


def _ean_only_record(text: str) -> MedicineRecord | None:
    digits = re.sub(r"\D", "", text)
    if len(digits) in (8, 12, 13, 14):
        gtin = normalize_gtin(digits)
        if gtin:
            return MedicineRecord(
                gtin=gtin,
                extraction_method="ean_barcode",
                confidence=0.6,
                raw_barcode=text,
                source_fields={"gtin": "barcode"},
            )
    return None


def decode_barcodes(image_bgr: np.ndarray) -> list[MedicineRecord]:
    if not HAS_ZXING:
        return []

    seen: set[str] = set()
    records: list[MedicineRecord] = []

    for variant in _preprocess_variants(image_bgr):
        for item in _decode_with_zxing(variant):
            text = item["text"].strip()
            if not text or text in seen:
                continue
            seen.add(text)

            if is_gs1_payload(text):
                record = parse_gs1(text)
                record.raw_barcode = text
                records.append(record)
                continue

            ean_record = _ean_only_record(text)
            if ean_record:
                records.append(ean_record)

    return records
