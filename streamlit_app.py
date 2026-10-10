import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Medicine Label Extractor", page_icon="💊", layout="wide")

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass

st.title("💊 Medicine Label Extractor")
st.caption("PaddleOCR-VL 1.6 → GTIN / Batch / Lot / MFG / EXP / Serial")

with st.sidebar:
    api_url = st.text_input("FastAPI URL", os.getenv("API_URL", "http://localhost:8000")).rstrip("/")
    debug = st.checkbox("Show raw OCR text", value=False)
    workers = st.slider("Parallel requests", 1, 8, 3)
    if st.button("Check API"):
        try:
            st.success(requests.get(f"{api_url}/health", timeout=10).json())
        except Exception as e:
            st.error(f"API unreachable: {e}")

files = st.file_uploader("Upload medicine images", type=["jpg", "jpeg", "png", "webp", "bmp"],
                         accept_multiple_files=True)


def call_api(name: str, data: bytes, mime: str) -> dict:
    try:
        r = requests.post(f"{api_url}/extract", params={"debug": str(debug).lower()},
                          files={"file": (name, data, mime)}, timeout=600)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"source_image": name, "medicines": [], "pipeline_steps": [], "errors": [str(e)]}


if files and st.button("Extract", type="primary"):
    payload = [(f.name, f.getvalue(), f.type or "image/jpeg") for f in files]
    results: dict[str, dict] = {}
    bar = st.progress(0.0, text="Processing…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call_api, *p): p[0] for p in payload}
        for i, fut in enumerate(as_completed(futs), 1):
            results[futs[fut]] = fut.result()
            bar.progress(i / len(futs), text=f"Processed {i}/{len(futs)}")
    bar.empty()
    st.session_state["results"] = [results[p[0]] for p in payload]
    st.session_state["images"] = {p[0]: p[1] for p in payload}

results = st.session_state.get("results")
if results:
    rows = []
    for r in results:
        for m in r["medicines"] or [{}]:
            rows.append({"source_image": r["source_image"], **{k: m.get(k) for k in (
                "gtin", "batch_no", "lot", "mfg_date", "exp_date", "serial_number",
                "extraction_method", "confidence")}, "errors": "; ".join(r["errors"])})
    df = pd.DataFrame(rows)
    st.subheader("Summary")
    st.dataframe(df, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download JSON", json.dumps(results, indent=2, ensure_ascii=False),
                       "extraction_results.json", "application/json")
    c2.download_button("Download CSV", df.to_csv(index=False), "extraction_results.csv", "text/csv")

    st.subheader("Per image")
    for r in results:
        ok = bool(r["medicines"])
        with st.expander(f"{'✅' if ok else '⚠️'} {r['source_image']}", expanded=False):
            left, right = st.columns([1, 2])
            img = st.session_state.get("images", {}).get(r["source_image"])
            if img:
                left.image(img, use_container_width=True)
            right.json({k: v for k, v in r.items() if k != "debug"})
            if r.get("debug"):
                right.caption(f"Rotation used: {r['debug'].get('rotation_used')}°")
                right.text_area("Raw OCR text", r["debug"].get("ocr_text", ""), height=200,
                                key=f"ocr_{r['source_image']}")
