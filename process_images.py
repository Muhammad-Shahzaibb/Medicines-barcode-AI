#!/usr/bin/env python3
"""
Batch extraction script.
Processes medicine images using the developed PaddleOCR-VL extraction pipeline
and saves individual JSON output files named after each image.

Example:
    python process_images.py
    python process_images.py --input-dir "Priority Test Data" --output-dir "output"
    python process_images.py --overwrite --concurrency 4
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.ocr_client import OCRClient
from app.pipeline import ExtractionPipeline

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def find_images(input_dir: Path) -> list[Path]:
    """Find all image files in input directory (skipping videos, json, etc.)."""
    images = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images


async def process_single_image(
    pipeline: ExtractionPipeline,
    image_path: Path,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    overwrite: bool,
    include_debug: bool,
    progress_counter: dict,
    total_images: int,
) -> dict:
    """Process a single image and write {image_stem}.json to output_dir."""
    output_path = output_dir / f"{image_path.stem}.json"

    # Skip existing if not overwriting
    if output_path.exists() and not overwrite:
        progress_counter["completed"] += 1
        idx = progress_counter["completed"]
        print(f"[{idx}/{total_images}] SKIP (already exists): {image_path.name}")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    async with semaphore:
        progress_counter["completed"] += 1
        idx = progress_counter["completed"]
        print(f"[{idx}/{total_images}] Processing: {image_path.name}...")

        try:
            data = image_path.read_bytes()
            res = await pipeline.process(image_path.name, data, debug=include_debug)
            res_dict = res.model_dump()
            if not include_debug:
                res_dict.pop("debug", None)
        except Exception as e:
            res_dict = {
                "source_image": image_path.name,
                "medicines": [],
                "pipeline_steps": [],
                "errors": [f"Processing error: {str(e)}"],
            }

        # Save to individual JSON file with same name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=2, ensure_ascii=False)

        med_count = len(res_dict.get("medicines", []))
        if med_count > 0:
            first_med = res_dict["medicines"][0]
            summary_info = []
            if first_med.get("batch_no"):
                summary_info.append(f"Batch: {first_med['batch_no']}")
            if first_med.get("lot"):
                summary_info.append(f"Lot: {first_med['lot']}")
            if first_med.get("exp_date"):
                summary_info.append(f"Exp: {first_med['exp_date']}")
            if first_med.get("gtin"):
                summary_info.append(f"GTIN: {first_med['gtin']}")
            info_str = ", ".join(summary_info) if summary_info else f"{med_count} medicine(s)"
            print(f"    -> Saved {output_path.name} [OK: {info_str}]")
        else:
            errs = "; ".join(res_dict.get("errors", []))
            print(f"    -> Saved {output_path.name} [WARN: No medicines extracted - {errs}]")

        return res_dict


async def run_batch(args: argparse.Namespace):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(input_dir)
    if not images:
        print(f"No supported images found in: {input_dir}")
        return

    print("=" * 70)
    print("MEDICINE LABEL EXTRACTION - BATCH PROCESSOR")
    print(f"Input directory : {input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Total images    : {len(images)}")
    print(f"Concurrency     : {args.concurrency}")
    print(f"Overwrite       : {args.overwrite}")
    print("=" * 70)

    settings = get_settings()
    ocr = OCRClient(settings)
    pipeline = ExtractionPipeline(ocr, settings)
    semaphore = asyncio.Semaphore(args.concurrency)

    progress_counter = {"completed": 0}

    try:
        tasks = [
            process_single_image(
                pipeline=pipeline,
                image_path=img,
                output_dir=output_dir,
                semaphore=semaphore,
                overwrite=args.overwrite,
                include_debug=args.include_debug,
                progress_counter=progress_counter,
                total_images=len(images),
            )
            for img in images
        ]
        all_results = await asyncio.gather(*tasks)

        # Write overall summary JSON file
        summary_path = output_dir / "_all_extractions_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        total_extracted = sum(1 for r in all_results if r.get("medicines"))
        print("=" * 70)
        print("BATCH PROCESSING COMPLETE!")
        print(f"Successfully processed: {len(all_results)} images")
        print(f"Images with extracted medicines: {total_extracted}/{len(all_results)}")
        print(f"Individual JSON files saved in: {output_dir.resolve()}")
        print(f"Consolidated summary saved to : {summary_path.resolve()}")
        print("=" * 70)

    finally:
        await ocr.aclose()


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Extract medicine information from images and save per-image JSON files."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default="Priority Test Data",
        help="Directory containing images to process (default: 'Priority Test Data').",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory where JSON files will be saved (default: 'output').",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=settings.max_concurrency or 4,
        help=f"Number of parallel requests to OCR service (default: {settings.max_concurrency or 4}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process and overwrite existing JSON files if they already exist.",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Include debug info (rotation angle, raw OCR text) in output JSON.",
    )

    args = parser.parse_args()
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
