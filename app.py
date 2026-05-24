import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Road Damage Detector",
    page_icon="🛣️",
    layout="wide"
)

# Title
st.title("Road Damage Detection System")
st.markdown("Upload a road photo and AI will detect damage instantly!")

# Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.9,
    value=0.10,
    step=0.05,
    help="Lower = detects more but less accurate. Higher = fewer but more confident detections."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Damage Classes:**")
st.sidebar.markdown("🔴 Longitudinal Crack")
st.sidebar.markdown("🟢 Transverse Crack")
st.sidebar.markdown("🔵 Alligator Crack")
st.sidebar.markdown("🟠 Pothole")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

model = load_model()

# Upload section
st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
    help="Upload any road photo — dashcam, drone, or phone camera"
)

if uploaded_file is not None:
    # Show original image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    # Run detection
    with st.spinner("Detecting road damage..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        # Run model
        results = model(tmp_path, conf=conf_threshold)
        result = results[0]

        # Draw boxes on image
        result_img = result.plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Clean up temp file
        os.unlink(tmp_path)

    with col2:
        st.subheader("Detection Result")
        st.image(result_img, use_column_width=True)

    # Stats
    st.markdown("---")
    st.subheader("Detection Summary")

    n_detections = len(result.boxes)

    if n_detections == 0:
        st.success("No road damage detected — this road looks clean!")
    else:
        st.warning(f"Found {n_detections} damage area(s) on this road!")

        # Class names
        class_names = {
            0: "Longitudinal Crack",
            1: "Transverse Crack",
            2: "Alligator Crack",
            3: "Pothole"
        }

        # Count each damage type
        class_counts = {}
        for box in result.boxes:
            cls = int(box.cls[0])
            name = class_names.get(cls, f"Class {cls}")
            conf = float(box.conf[0])
            if name not in class_counts:
                class_counts[name] = []
            class_counts[name].append(conf)

        # Display counts in columns
        cols = st.columns(len(class_counts))
        for i, (damage_type, confs) in enumerate(class_counts.items()):
            with cols[i]:
                avg_conf = sum(confs) / len(confs)
                st.metric(
                    label=damage_type,
                    value=f"{len(confs)} found",
                    delta=f"Avg confidence: {avg_conf:.0%}"
                )

        # Severity assessment
        st.markdown("---")
        st.subheader("Severity Assessment")
        if n_detections >= 5:
            st.error("SEVERE damage — Immediate repair recommended!")
        elif n_detections >= 3:
            st.warning("MODERATE damage — Schedule repair soon")
        elif n_detections >= 1:
            st.info("MINOR damage — Monitor regularly")

else:
    # Show instructions when no image uploaded
    st.info("Upload a road image above to start detection")
    st.markdown("""
    **How to use:**
    1. Click 'Browse files' above
    2. Select any road photo from your device
    3. Wait 2-3 seconds for AI detection
    4. View results with damage boxes and severity report

    **Best results with:**
    - Dashcam footage
    - Road-level photos
    - Clear daytime images
    """)

st.markdown("---")
st.caption("Road Damage Detection System — Built with YOLOv8 and Streamlit")
