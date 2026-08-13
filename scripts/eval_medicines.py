from __future__ import annotations

import json
import mimetypes
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.pipeline import process_image

DATA_DIR = ROOT / "Medicines data"
OUTPUT_DIR = ROOT / "output"
FIELDS = ("gtin", "batch_no", "lot", "mfg_date", "exp_date", "serial_number")


def _mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "image/jpeg"


def _public(record) -> dict:
    return {field: getattr(record, field) for field in FIELDS}


def main() -> None:
    images = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    if not images:
        raise SystemExit(f"No images found in {DATA_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []
    filled = Counter()
    errors = 0

    print(f"Evaluating {len(images)} images from {DATA_DIR.name}")
    for idx, path in enumerate(images, start=1):
        image_bytes = path.read_bytes()
        result = process_image(image_bytes, source_name=path.name, mime=_mime(path))
        row = {
            "image": path.name,
            "medicines": [_public(m) for m in result.medicines],
            "errors": result.errors,
            "pipeline_steps": result.pipeline_steps,
        }
        results.append(row)
        if result.errors:
            errors += 1
        for medicine in result.medicines:
            for field in FIELDS:
                if getattr(medicine, field):
                    filled[field] += 1
        summary = row["medicines"][0] if row["medicines"] else {"error": result.errors}
        print(f"[{idx:02d}/{len(images)}] {path.name}")
        print(json.dumps(summary, ensure_ascii=False))

    out_path = OUTPUT_DIR / "medicines_eval.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nFill counts (non-null fields):")
    for field in FIELDS:
        print(f"  {field}: {filled[field]}/{len(images)}")
    print(f"Images with errors: {errors}/{len(images)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
