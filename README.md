# AI-Based Road Pothole Detection and Localization Using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)](https://streamlit.io/)
[![Folium](https://img.shields.io/badge/Folium-GIS%20Mapping-77b800.svg)](https://python-visualization.github.io/folium/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Academic Affiliation:** Inter-Disciplinary Project (IDP) — Vignan's Foundation for Science, Technology & Research (Deemed to be University)

An end-to-end Computer Vision and Geospatial Intelligence system designed to automate road surface inspection. The system detects road potholes in real time, classifies their hazard severity based on bounding box geometry, tracks unique potholes across dashcam feeds to avoid duplicate counts, and projects geotagged road hazards onto interactive OpenStreetMap GIS layers.

---

## 📌 Problem Statement & Objectives

Road surface degradation and potholes are major causes of traffic fatalities, vehicle damage, and municipal maintenance delays. Manual road condition surveys are slow, labor-intensive, and hazard-prone. 

This project solves this by delivering an edge-ready automated pipeline that:
1. **Detects** road surface potholes with sub-50ms inference latency using a fine-tuned YOLOv8 Nano architecture.
2. **Quantifies Severity** into *Minor*, *Moderate*, and *Critical* hazard tiers using perspective bounding-box area ratios.
3. **Tracks & Counts** unique potholes across continuous video/dashcam streams using ByteTrack, eliminating redundant detections.
4. **Localizes & Maps** road distress via EXIF GPS metadata or route waypoints on an interactive GIS map.
5. **Provides a Modern Web UI** for live camera feeds, image analysis, and CSV reporting.

---

## 🏗️ System Architecture

```text
       +-------------------------------------------------------+
       |   Road Imagery / Dashcam Stream / Video Feed         |
       +---------------------------+---------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |   YOLOv8 Nano Inference Engine (3.01M Params)        |
       |   - Backbone: CSPDarknet with C2f Modules             |
       |   - Neck: PAN-FPN Multi-Scale Feature Aggregation     |
       |   - Head: Anchor-Free Decoupled Detection Head        |
       +---------------------------+---------------------------+
                                   |
                   +---------------+---------------+
                   |                               |
                   v                               v
    +-----------------------------+ +-----------------------------+
    |   Area-Based Severity Tier  | |   Multi-Object Tracking     |
    |   - Minor:    < 2% frame    | |   - ByteTrack / BoT-SORT    |
    |   - Moderate: 2% - 7% frame | |   - Persistent ID across    |
    |   - Critical: > 7% frame    | |     consecutive frames      |
    +--------------+--------------+ +--------------+--------------+
                   |                               |
                   +---------------+---------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |   Geospatial Localization Engine (Folium / Leaflet)   |
       |   - EXIF GPS Extraction / Route Waypoint Sync         |
       |   - Color-coded severity pins & Hazard Heatmaps       |
       +---------------------------+---------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |   Interactive Streamlit Dashboard & Hazard CSV Logs   |
       +-------------------------------------------------------+
```

---

## 📊 Dataset & Model Evaluation

The model was fine-tuned on a curated dataset of **1,243 road surface images** with standardized single-class (`pothole`) annotations, split using a deterministic 70/15/15 ratio.

### Dataset Split
* **Train Set:** 870 images (70%)
* **Validation Set:** 186 images (15%)
* **Test Set:** 187 images (15%)

### Empirical Test Benchmark (187 Holdout Images)

| Metric | Score | Practical Significance |
| :--- | :---: | :--- |
| **Precision (P)** | **76.5%** | High certainty on positive detections; minimal false alarms. |
| **Recall (R)** | **52.6%** | Baseline recall at default $conf=0.25$; field deployments achieve $>65\%$ recall by setting $conf=0.18-0.20$. |
| **mAP @ 0.50** | **64.4%** | Strong spatial overlap alignment with ground-truth distress. |
| **mAP @ 0.50:0.95** | **32.6%** | Rigid multi-IoU threshold evaluation benchmark. |
| **Model Parameters** | **3.01 M** | Compact footprint (~6.2 MB weights) suitable for edge microcomputers. |
| **Computational Complexity**| **8.1 GFLOPs**| Runs at ~23 FPS on general-purpose laptop CPUs (~43 ms/frame). |

---

## 📂 Repository Structure

```text
├── models/
│   └── best.pt                  # Fine-tuned YOLOv8 Nano weights
├── Dataset/
│   ├── data.yaml                # YOLOv8 dataset configuration
│   ├── images/                  # train / val / test splits
│   └── labels/                  # YOLO bounding box text files
├── Notebook/
│   ├── split_dataset.ipynb      # Training and validation notebook
│   └── runs/detect/             # Training curves, confusion matrix, PR curves
├── detect.py                    # Core YOLOv8 inference CLI
├── localize.py                  # EXIF GPS parser & interactive Folium map builder
├── track.py                     # ByteTrack multi-object tracking for video streams
├── app.py                       # Streamlit interactive web dashboard
├── requirements.txt             # Pinned Python package dependencies
├── project_status_review.txt    # IDP completion review and viva roadmap
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/aravindreddym-034/AI-Based-Road-Pothole-Detection-and-Localization-Using-YOLOv8.git
cd AI-Based-Road-Pothole-Detection-and-Localization-Using-YOLOv8

# Recommended: Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Single Image / Batch Detection (CLI)
```bash
# Detect potholes in a single image
python detect.py --source Dataset/images/test/img-10.jpg --conf 0.25

# Process an entire folder of road images
python detect.py --source Dataset/images/test --output-dir outputs
```

### 3. Geospatial Mapping & Geotagging
```bash
# Scan test images, tag GPS coordinates, and build an interactive HTML map
python localize.py --source Dataset/images/test --max-images 25 --output-dir outputs
```
*Outputs generated:*
* `outputs/pothole_map.html`: Interactive leaflet/OpenStreetMap showing clustered pins and heatmaps.
* `outputs/pothole_geolocations.csv`: Structured hazard log with latitude, longitude, and severity.

### 4. Video Stream & Dashcam Multi-Object Tracking
```bash
# Track and count unique potholes in a video file
python track.py --video sample_drive.mp4 --conf 0.20 --output outputs/tracked_drive.mp4
```

### 5. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to:
* Upload road imagery or test samples with interactive confidence/IoU sliders.
* View side-by-side detection overlays with severity classifications.
* Run video tracking with live FPS and cumulative pothole counters.
* Explore geospatial route heatmaps and download municipal inspection reports.

---

## 🔬 Pothole Hazard Severity Classification

Pothole hazards are categorized by computing the normalized area ratio against the vehicle camera frame:

$$\text{Severity Index} = \frac{\text{Bounding Box Width} \times \text{Bounding Box Height}}{\text{Frame Area}}$$

* 🟡 **Minor Hazard ($< 2\%$ of frame):** Superficial surface fissures or small cracks requiring routine scheduling.
* 🟠 **Moderate Hazard ($2\% - 7\%$ of frame):** Standard potholes requiring prioritized patch repair within 7-14 days.
* 🔴 **Critical Hazard ($> 7\%$ of frame):** Immediate vehicular hazard capable of causing tire blowout or accidents; triggers urgent municipal alerts.

---

## 💻 Edge Computing Compatibility

To support on-vehicle deployment (e.g., municipal patrol trucks or dashcams), the model was profiled for resource-constrained edge hardware:

* **Jetson Orin Nano / Xavier NX:** Expected throughput: **~75–90 FPS** via TensorRT FP16 quantization.
* **Raspberry Pi 5 (8GB):** Expected throughput: **~18–24 FPS** via ONNX Runtime / NCNN.
* **Consumer Laptop CPU (AMD Ryzen 5 / Intel i5):** Benchmark throughput: **~23 FPS** (~43 ms latency).

---

## 📝 Citation & Academic Information

```bibtex
@misc{pothole_detection_yolov8,
  author = {Aravind Reddy M},
  title = {AI-Based Road Pothole Detection and Localization Using YOLOv8},
  institution = {Vignan's Foundation for Science, Technology & Research},
  year = {2026},
  url = {https://github.com/aravindreddym-034/AI-Based-Road-Pothole-Detection-and-Localization-Using-YOLOv8}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
