"""
app.py - AI-Based Road Pothole Detection & Localization Interactive Dashboard
Web application powered by Streamlit, YOLOv8, and Folium.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2
import folium
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_folium import st_folium

from detect import (
    PotholeDetector,
    DEFAULT_MODEL_PATH,
    classify_severity,
    SEVERITY_COLORS
)
from localize import PotholeLocalizer, extract_exif_gps, DEFAULT_BASE_LAT, DEFAULT_BASE_LON

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Road Pothole Detection & Localization",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-moderate {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-minor {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Cache the YOLOv8 model in memory."""
    return PotholeDetector(model_path=DEFAULT_MODEL_PATH)


detector = load_detector()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/road.png", width=64)
st.sidebar.title("Configuration")
st.sidebar.markdown("**IDP Capstone Project**  \n*Vignan's University*")

conf_slider = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
    help="Lower threshold improves recall for road hazards; higher threshold reduces false positives."
)

iou_slider = st.sidebar.slider(
    "NMS IoU Threshold",
    min_value=0.10,
    max_value=0.80,
    value=0.45,
    step=0.05
)

st.sidebar.divider()
st.sidebar.markdown("### 📍 Location Anchor")
sim_lat = st.sidebar.number_input("Base Latitude", value=DEFAULT_BASE_LAT, format="%.5f")
sim_lon = st.sidebar.number_input("Base Longitude", value=DEFAULT_BASE_LON, format="%.5f")

# Navigation Tabs
tab_image, tab_video, tab_map, tab_analytics = st.tabs([
    "📸 Image Inspection",
    "🎥 Video Dashcam",
    "🗺️ Geospatial Map",
    "📊 Model Performance"
])

# -------------------------------------------------------------
# TAB 1: IMAGE INSPECTION
# -------------------------------------------------------------
with tab_image:
    st.markdown('<div class="main-header">Road Pothole Detection & Geotagging</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a road surface image or pick a test sample to identify and categorize potholes.</div>', unsafe_allow_html=True)

    test_images_dir = Path("Dataset/images/test")
    sample_files = []
    if test_images_dir.exists():
        sample_files = [f.name for f in test_images_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]][:15]

    col_input_type, col_select = st.columns([1, 2])
    with col_input_type:
        input_mode = st.radio("Input Source:", ["Pick Test Sample", "Upload Image"], horizontal=True)

    input_img = None
    image_name = "sample.jpg"

    if input_mode == "Pick Test Sample" and sample_files:
        with col_select:
            selected_sample = st.selectbox("Choose sample road image:", sample_files)
        sample_path = test_images_dir / selected_sample
        input_img = Image.open(sample_path)
        image_name = selected_sample
    elif input_mode == "Upload Image":
        with col_select:
            uploaded_file = st.file_uploader("Upload road image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            input_img = Image.open(uploaded_file)
            image_name = uploaded_file.name

    if input_img is not None:
        cv_img = np.array(input_img.convert("RGB"))
        cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing road surface..."):
            res = detector.predict_image(cv_img_bgr, conf_threshold=conf_slider, iou_threshold=iou_slider)

        annotated_rgb = cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB)
        count = res["pothole_count"]

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Potholes Detected", count)
        with m2:
            st.metric("Inference Latency", f"{res['latency_ms']} ms")
        with m3:
            severities = [d["severity"] for d in res["detections"]]
            max_sev = "Critical" if "Critical" in severities else ("Moderate" if "Moderate" in severities else ("Minor" if count > 0 else "None"))
            st.metric("Peak Hazard Level", max_sev)
        with m4:
            avg_conf = f"{np.mean([d['confidence'] for d in res['detections']]) * 100:.1f}%" if count > 0 else "N/A"
            st.metric("Mean Confidence", avg_conf)

        st.divider()

        # Side by side visual
        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.subheader("Original Input")
            st.image(input_img, use_container_width=True)
        with col_annot:
            st.subheader(f"YOLOv8 Detection ({count} Hazards)")
            st.image(annotated_rgb, use_container_width=True)

        if count > 0:
            st.subheader("Detected Pothole Breakdown")
            df = pd.DataFrame(res["detections"])
            display_df = df[["id", "severity", "confidence", "area_ratio", "bbox"]].copy()
            display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
            display_df["area_ratio"] = (display_df["area_ratio"] * 100).round(2).astype(str) + "% of frame"
            display_df.columns = ["Hazard ID", "Severity", "Confidence", "Relative Size", "Bounding Box (XYXY)"]
            st.dataframe(display_df, use_container_width=True)

            # Interactive Single-Pin Folium Map
            st.subheader("📍 Geospatial Localization Pin")
            single_map = folium.Map(location=[sim_lat, sim_lon], zoom_start=15)
            marker_color = "red" if max_sev == "Critical" else ("orange" if max_sev == "Moderate" else "green")
            folium.Marker(
                location=[sim_lat, sim_lon],
                popup=f"<b>{image_name}</b><br>{count} pothole(s) detected.<br>Severity: {max_sev}",
                tooltip=f"Road Hazard: {count} Potholes",
                icon=folium.Icon(color=marker_color, icon="warning-sign")
            ).add_to(single_map)
            st_folium(single_map, width="100%", height=320)
        else:
            st.success("✅ Clean road surface detected! No potholes identified above confidence threshold.")

# -------------------------------------------------------------
# TAB 2: VIDEO DASHCAM STREAM
# -------------------------------------------------------------
with tab_video:
    st.markdown('<div class="main-header">Real-Time Video & Dashcam Tracking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Feed continuous vehicle video to track pothole IDs and eliminate duplicate counts across frames.</div>', unsafe_allow_html=True)

    uploaded_vid = st.file_uploader("Upload driving/dashcam video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"])

    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile_path = tfile.name

        st.video(tfile_path)

        if st.button("🚀 Run Multi-Object Pothole Tracking", type="primary"):
            from track import PotholeTracker
            tracker = PotholeTracker()

            output_vid_path = os.path.join("outputs", f"tracked_{Path(uploaded_vid.name).stem}.mp4")
            with st.spinner("Processing video frames with ByteTrack..."):
                t_res = tracker.process_video(
                    video_path=tfile_path,
                    output_path=output_vid_path,
                    conf_threshold=conf_slider
                )

            st.success(f"Processing Complete! Identified **{t_res['unique_potholes_count']} unique potholes** across {t_res['total_frames']} frames ({t_res['average_fps']} FPS).")

            if os.path.exists(output_vid_path):
                st.subheader("Processed Video Output")
                st.video(output_vid_path)

# -------------------------------------------------------------
# TAB 3: GEOSPATIAL MAP
# -------------------------------------------------------------
with tab_map:
    st.markdown('<div class="main-header">Geospatial Hazard Density Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive route visualization plotting detected road distress hotspots.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("#### Inspection Batch")
        batch_count = st.slider("Number of Test Images to Scan", min_value=5, max_value=50, value=25)
        run_scan = st.button("🗺️ Scan & Generate Map", type="primary")

    if run_scan:
        localizer = PotholeLocalizer(detector=detector)
        test_dir = "Dataset/images/test"
        if os.path.exists(test_dir):
            with st.spinner("Running batch localization and generating OpenStreetMap layers..."):
                localizer.process_directory(test_dir, conf_threshold=conf_slider, max_images=batch_count)
                html_map_path = "outputs/pothole_map.html"
                csv_map_path = "outputs/pothole_geolocations.csv"
                folium_map = localizer.build_map(output_html=html_map_path)
                localizer.export_csv(output_csv=csv_map_path)

            st.session_state["cached_map"] = folium_map
            st.session_state["cached_records"] = localizer.records

    if "cached_map" in st.session_state:
        st_folium(st.session_state["cached_map"], width="100%", height=520)

        if "cached_records" in st.session_state and st.session_state["cached_records"]:
            df_recs = pd.DataFrame(st.session_state["cached_records"])
            st.subheader(f"Route Hazard Log ({len(df_recs)} Records)")
            st.dataframe(df_recs[["image_name", "latitude", "longitude", "severity", "confidence", "area_ratio"]], use_container_width=True)

            csv_data = df_recs.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Hazard Log (CSV)",
                data=csv_data,
                file_name="pothole_hazard_log.csv",
                mime="text/csv"
            )
    else:
        st.info("Click **Scan & Generate Map** to build an interactive GIS heatmap from road inspection images.")

# -------------------------------------------------------------
# TAB 4: MODEL PERFORMANCE & ANALYTICS
# -------------------------------------------------------------
with tab_analytics:
    st.markdown('<div class="main-header">YOLOv8 Training & Test Evaluation Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical validation metrics on the 187-image holdout test dataset.</div>', unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Precision (P)", "76.5%", help="Accuracy of detected bounding boxes")
    with b2:
        st.metric("Recall (R)", "52.6%", help="Proportion of actual potholes identified")
    with b3:
        st.metric("mAP @ 0.50", "64.4%", help="Mean Average Precision at IoU=0.5")
    with b4:
        st.metric("mAP @ 0.50:0.95", "32.6%", help="Strict Mean Average Precision across all IoU thresholds")

    st.divider()

    runs_dir = Path("Notebook/runs/detect/pothole_yolov8_baseline-2")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Confusion Matrix")
        cf_path = runs_dir / "confusion_matrix.png"
        if cf_path.exists():
            st.image(str(cf_path), caption="Normalized Confusion Matrix", use_container_width=True)
        else:
            st.write("Confusion matrix not found.")

        st.subheader("Precision-Recall (PR) Curve")
        pr_path = runs_dir / "BoxPR_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="PR Curve (mAP@0.50 = 0.644)", use_container_width=True)

    with col_p2:
        st.subheader("F1-Confidence Curve")
        f1_path = runs_dir / "BoxF1_curve.png"
        if f1_path.exists():
            st.image(str(f1_path), caption="F1 Curve vs Confidence Threshold", use_container_width=True)

        st.subheader("Training Loss Convergence")
        res_path = runs_dir / "results.png"
        if res_path.exists():
            st.image(str(res_path), caption="Training & Validation Losses over 50 Epochs", use_container_width=True)

    st.markdown("""
    ### 💡 Technical Analysis & Viva Summary
    * **Model Choice:** YOLOv8 Nano (`yolov8n.pt`) with **3.01M parameters** and **8.1 GFLOPs**, specifically chosen for edge compatibility on resource-constrained vehicle hardware.
    * **Precision vs Recall Tradeoff:** High precision (76.5%) ensures few false alarms. Recall (52.6%) can be effectively augmented in field deployments by operating at a lower confidence threshold ($conf=0.18-0.20$).
    * **Inference Speed:** Average latency of **~43 ms/frame** enables real-time throughput of **23+ FPS** on standard CPU hardware without requiring dedicated discrete GPUs.
    """)
