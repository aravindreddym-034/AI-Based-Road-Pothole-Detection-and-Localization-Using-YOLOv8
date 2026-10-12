"""
localize.py - Road Pothole Geospatial Localization Engine
Extracts image GPS EXIF metadata (or generates route waypoint geotags)
and maps detected potholes onto interactive OpenStreetMap layers with severity indicators.
"""

import argparse
import base64
import csv
import io
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from detect import PotholeDetector, DEFAULT_MODEL_PATH


# Default anchor coordinates (Vignan University / Guntur-Tenali Highway corridor)
DEFAULT_BASE_LAT = 16.2334
DEFAULT_BASE_LON = 80.5487


def get_decimal_from_dms(dms, ref):
    """Convert degrees, minutes, seconds tuple to decimal degrees."""
    degrees = float(dms[0])
    minutes = float(dms[1]) / 60.0
    seconds = float(dms[2]) / 3600.0
    decimal = degrees + minutes + seconds
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal


def extract_exif_gps(image_path: str) -> Optional[Tuple[float, float]]:
    """
    Extracts latitude and longitude from image EXIF metadata if present.
    Returns (lat, lon) or None if no GPS metadata exists.
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None

            gps_info = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    for key in value:
                        sub_tag = GPSTAGS.get(key, key)
                        gps_info[sub_tag] = value[key]

            if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                lat = get_decimal_from_dms(
                    gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N")
                )
                lon = get_decimal_from_dms(
                    gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E")
                )
                return (round(lat, 6), round(lon, 6))
    except Exception:
        pass
    return None


def generate_simulated_route_coords(
    index: int,
    total: int,
    base_lat: float = DEFAULT_BASE_LAT,
    base_lon: float = DEFAULT_BASE_LON,
    jitter: float = 0.0003
) -> Tuple[float, float]:
    """
    Generates realistic linear waypoint coordinates simulating vehicle progression
    along a road corridor when EXIF data is absent.
    """
    step = index / max(1, total)
    # Travel along road heading North-East
    lat = base_lat + (step * 0.015) + (random.uniform(-jitter, jitter))
    lon = base_lon + (step * 0.012) + (random.uniform(-jitter, jitter))
    return (round(lat, 6), round(lon, 6))


class PotholeLocalizer:
    """Geospatial Mapping & Localization Pipeline for Road Distress."""

    def __init__(self, detector: Optional[PotholeDetector] = None, model_path: str = DEFAULT_MODEL_PATH):
        self.detector = detector or PotholeDetector(model_path=model_path)
        self.records: List[Dict] = []

    def process_image(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        custom_lat: Optional[float] = None,
        custom_lon: Optional[float] = None,
        index: int = 0,
        total: int = 1
    ) -> Dict:
        """
        Runs detection and tags geographic coordinates.
        """
        res = self.detector.predict_image(image_path, conf_threshold=conf_threshold)

        # GPS resolution order: 1) Custom args -> 2) EXIF -> 3) Simulated Route
        gps = None
        if custom_lat is not None and custom_lon is not None:
            gps = (custom_lat, custom_lon)
            gps_source = "User Supplied"
        else:
            gps = extract_exif_gps(image_path)
            if gps:
                gps_source = "EXIF Metadata"
            else:
                gps = generate_simulated_route_coords(index, total)
                gps_source = "Route Waypoint (Estimated)"

        lat, lon = gps
        pothole_items = []

        for d in res["detections"]:
            item = {
                "image_name": Path(image_path).name,
                "image_path": str(image_path),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pothole_id": d["id"],
                "latitude": lat,
                "longitude": lon,
                "gps_source": gps_source,
                "severity": d["severity"],
                "confidence": d["confidence"],
                "area_ratio": d["area_ratio"],
                "bbox": d["bbox"]
            }
            self.records.append(item)
            pothole_items.append(item)

        return {
            "image_name": Path(image_path).name,
            "latitude": lat,
            "longitude": lon,
            "gps_source": gps_source,
            "pothole_count": res["pothole_count"],
            "detections": pothole_items,
            "annotated_image": res["annotated_image"]
        }

    def process_directory(
        self,
        directory_path: str,
        conf_threshold: float = 0.25,
        max_images: Optional[int] = None
    ) -> List[Dict]:
        """Process an entire directory of road images."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [
            f for f in Path(directory_path).iterdir()
            if f.suffix.lower() in valid_exts
        ]
        if max_images:
            files = files[:max_images]

        total = len(files)
        results = []
        print(f"[LOCALIZE] Processing {total} images from {directory_path}...")

        for i, file_path in enumerate(files):
            r = self.process_image(str(file_path), conf_threshold=conf_threshold, index=i, total=total)
            results.append(r)

        return results

    def build_map(
        self,
        output_html: str = "outputs/pothole_map.html",
        include_heatmap: bool = True
    ) -> folium.Map:
        """
        Build an interactive Folium map with clustering, color-coded pins, and popups.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)

        if not self.records:
            center_lat, center_lon = DEFAULT_BASE_LAT, DEFAULT_BASE_LON
        else:
            center_lat = sum(r["latitude"] for r in self.records) / len(self.records)
            center_lon = sum(r["longitude"] for r in self.records) / len(self.records)

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="OpenStreetMap"
        )

        # Tile layer options (OpenStreetMap default)

        # Marker Cluster
        marker_cluster = MarkerCluster(name="Clustered Potholes").add_to(m)

        color_map = {
            "Minor": "orange",
            "Moderate": "darkred",
            "Critical": "red"
        }
        icon_map = {
            "Minor": "info-sign",
            "Moderate": "warning-sign",
            "Critical": "remove-sign"
        }

        heat_data = []

        for r in self.records:
            lat, lon = r["latitude"], r["longitude"]
            severity = r["severity"]
            conf = r["confidence"]
            heat_weight = 1.0 if severity == "Critical" else (0.6 if severity == "Moderate" else 0.3)
            heat_data.append([lat, lon, heat_weight])

            # HTML Popup
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 220px; font-size: 13px;">
                <h4 style="margin: 0 0 6px 0; color: #1f2937;">Pothole #{r['pothole_id']}</h4>
                <div style="margin-bottom: 4px;">
                    <b>Severity:</b> 
                    <span style="color: {'#dc2626' if severity=='Critical' else '#d97706'}; font-weight: bold;">
                        {severity}
                    </span>
                </div>
                <div><b>Confidence:</b> {conf * 100:.1f}%</div>
                <div><b>Area Ratio:</b> {r['area_ratio'] * 100:.2f}%</div>
                <div><b>GPS:</b> {lat:.5f}, {lon:.5f}</div>
                <div style="color: #6b7280; font-size: 11px; margin-top: 4px;">
                    {r['image_name']} ({r['gps_source']})
                </div>
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{severity} Pothole ({conf*100:.0f}%)",
                icon=folium.Icon(
                    color=color_map.get(severity, "red"),
                    icon=icon_map.get(severity, "warning-sign"),
                    prefix="glyphicon"
                )
            ).add_to(marker_cluster)

        # Heatmap Layer
        if include_heatmap and heat_data:
            HeatMap(heat_data, name="Pothole Heatmap (Density)", radius=18, blur=12).add_to(m)

        folium.LayerControl().add_to(m)
        m.save(output_html)
        print(f"[MAP SAVED] Interactive Map written to: {output_html}")
        return m

    def export_csv(self, output_csv: str = "outputs/pothole_geolocations.csv"):
        """Exports all localized pothole records to a CSV log."""
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        if not self.records:
            print("[WARN] No records to export to CSV.")
            return

        keys = [
            "image_name", "timestamp", "pothole_id", "latitude", "longitude",
            "gps_source", "severity", "confidence", "area_ratio"
        ]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.records:
                filtered_row = {k: r.get(k, "") for k in keys}
                writer.writerow(filtered_row)

        print(f"[CSV SAVED] Pothole Log exported to: {output_csv} ({len(self.records)} records)")


def main():
    parser = argparse.ArgumentParser(description="Pothole Localization & Mapping Engine")
    parser.add_argument("--source", type=str, default="Dataset/images/test", help="Path to image or directory.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to YOLOv8 weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-images", type=int, default=30, help="Max images to process in directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for map and reports.")
    args = parser.parse_args()

    localizer = PotholeLocalizer(model_path=args.model)
    src = Path(args.source)

    if src.is_file():
        localizer.process_image(str(src), conf_threshold=args.conf)
    elif src.is_dir():
        localizer.process_directory(str(src), conf_threshold=args.conf, max_images=args.max_images)
    else:
        print(f"[ERROR] Source not found: {args.source}")
        return

    map_path = os.path.join(args.output_dir, "pothole_map.html")
    csv_path = os.path.join(args.output_dir, "pothole_geolocations.csv")

    localizer.build_map(output_html=map_path)
    localizer.export_csv(output_csv=csv_path)


if __name__ == "__main__":
    main()
