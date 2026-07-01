import json
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.pipeline import process_image

load_dotenv()

st.set_page_config(
    page_title="Medicine Barcode Extractor",
    page_icon="💊",
    layout="wide",
)

st.title("Medicine Barcode Extractor")
st.caption("Vision AI extraction via Groq Llama 4 Scout")

with st.sidebar:
    st.header("Settings")
    if not os.getenv("GROQ_API_KEY"):
        st.error("Set GROQ_API_KEY in your .env file.")
    else:
        st.success("GROQ_API_KEY loaded")

uploaded = st.file_uploader(
    "Upload medicine packaging image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded:
    image_bytes = uploaded.getvalue()
    mime = uploaded.type or "image/jpeg"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        st.image(Image.open(BytesIO(image_bytes)), width="stretch")

    with col2:
        st.subheader("Output")

        with st.spinner("Analyzing image with vision AI..."):
            try:
                result = process_image(
                    image_bytes=image_bytes,
                    source_name=uploaded.name,
                    mime=mime,
                )
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.stop()

        if result.errors:
            for err in result.errors:
                st.warning(err)

        if result.medicines:
            st.success(f"Found {len(result.medicines)} medicine record(s)")
            for idx, medicine in enumerate(result.medicines, start=1):
                with st.expander(f"Medicine {idx}", expanded=True):
                    output = {
                        "gtin": medicine.gtin,
                        "batch_no": medicine.batch_no,
                        "lot": medicine.lot,
                        "mfg_date": medicine.mfg_date,
                        "exp_date": medicine.exp_date,
                        "serial_number": medicine.serial_number,
                    }
                    st.json(output)
        else:
            st.info("No records extracted.")

        st.download_button(
            "Download JSON",
            data=json.dumps(result.to_dict(), indent=2),
            file_name=f"{os.path.splitext(uploaded.name)[0]}_medicines.json",
            mime="application/json",
        )
else:
    st.info("Upload a medicine label image to extract traceability data.")
