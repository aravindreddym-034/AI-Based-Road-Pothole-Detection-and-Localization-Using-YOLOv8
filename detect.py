"""
detect.py - Road Pothole Detection CLI & Core Engine
Uses YOLOv8 Nano fine-tuned on road pothole imagery.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

# Default model path
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "best.pt"
)

# Severity Configuration based on normalized bounding box area
SEVERITY_THRESHOLDS = {
    "minor": 0.02,     # < 2% of frame area
    "moderate": 0.07,  # 2% - 7% of frame area
    "critical": 1.00   # > 7% of frame area
}

SEVERITY_COLORS = {
    "Minor": (0, 215, 255),      # Amber/Yellow (BGR)
    "Moderate": (0, 140, 255),   # Orange (BGR)
    "Critical": (0, 0, 255)      # Red (BGR)
}


def classify_severity(bbox_area_ratio: float) -> str:
    """Classifies pothole hazard severity based on frame area ratio."""
    if bbox_area_ratio < SEVERITY_THRESHOLDS["minor"]:
        return "Minor"
    elif bbox_area_ratio < SEVERITY_THRESHOLDS["moderate"]:
        return "Moderate"
    else:
        return "Critical"


class PotholeDetector:
    """YOLOv8-based Road Pothole Detection & Analysis Engine."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = ""):
        if not os.path.exists(model_path):
            # Fallback check inside Notebook runs if models/best.pt doesn't exist
            fallback = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Notebook", "runs", "detect", "pothole_yolov8_baseline-2", "weights", "best.pt"
            )
            if os.path.exists(fallback):
                model_path = fallback
            else:
                raise FileNotFoundError(f"Model weights not found at: {model_path}")

        self.model_path = model_path
        self.model = YOLO(model_path)
        self.device = device
        print(f"[INFO] Loaded Pothole Detection Model: {model_path}")

    def predict_image(
        self,
        image_input: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict:
        """
        Run inference on a single image.
        Returns a dict with detections, severity ratings, latency, and annotated image.
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not read image: {image_input}")
            img_name = Path(image_input).name
        else:
            image = image_input.copy()
            img_name = "in_memory_image"

        img_h, img_w = image.shape[:2]
        frame_area = img_h * img_w

        t_start = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        inference_latency_ms = (time.perf_counter() - t_start) * 1000

        detections = []
        annotated_image = image.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                box_w = max(0, x2 - x1)
                box_h = max(0, y2 - y1)
                box_area = box_w * box_h
                area_ratio = box_area / frame_area if frame_area > 0 else 0.0

                severity = classify_severity(area_ratio)
                color = SEVERITY_COLORS.get(severity, (0, 0, 255))

                detections.append({
                    "id": i + 1,
                    "bbox": [x1, y1, x2, y2],
                    "width": box_w,
                    "height": box_h,
                    "area_ratio": round(area_ratio, 4),
                    "confidence": round(float(conf), 4),
                    "severity": severity,
                    "class_name": self.model.names.get(cls_id, "pothole")
                })

                # Visual annotation
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                label = f"{severity} ({conf:.2f})"
                (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    annotated_image,
                    (x1, max(0, y1 - lbl_h - 6)),
                    (x1 + lbl_w + 6, y1),
                    color,
                    -1
                )
                cv2.putText(
                    annotated_image,
                    label,
                    (x1 + 3, max(lbl_h + 2, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return {
            "image_name": img_name,
            "width": img_w,
            "height": img_h,
            "pothole_count": len(detections),
            "detections": detections,
            "latency_ms": round(inference_latency_ms, 2),
            "annotated_image": annotated_image
        }

    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        display: bool = False
    ) -> Dict:
        """
        Run inference frame-by-frame on a video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_detections = 0
        frame_idx = 0
        start_time = time.time()

        print(f"[INFO] Processing video: {video_path} ({total_frames} frames @ {fps:.1f} FPS)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = self.predict_image(frame, conf_threshold=conf_threshold)
            annotated_frame = result["annotated_image"]
            total_detections += result["pothole_count"]

            # Overlay summary HUD
            fps_current = (frame_idx + 1) / max(0.001, (time.time() - start_time))
            hud_text = f"Frame: {frame_idx}/{total_frames} | Potholes: {result['pothole_count']} | FPS: {fps_current:.1f}"
            cv2.putText(
                annotated_frame, hud_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )

            if writer:
                writer.write(annotated_frame)

            if display:
                cv2.imshow("Road Pothole Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        avg_fps = frame_idx / max(0.001, elapsed)

        print(f"[INFO] Completed video processing in {elapsed:.2f}s ({avg_fps:.1f} avg FPS)")
        return {
            "total_frames": frame_idx,
            "total_detections": total_detections,
            "average_fps": round(avg_fps, 2),
            "output_video": output_path
        }


def main():
    parser = argparse.ArgumentParser(description="AI Road Pothole Detection CLI (YOLOv8)")
    parser.add_argument("--source", type=str, required=True, help="Path to input image, directory, or video file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to YOLOv8 weights (.pt).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45).")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save prediction results.")
    parser.add_argument("--view", action="store_true", help="Display live prediction window.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    detector = PotholeDetector(model_path=args.model)

    source_path = Path(args.source)

    if source_path.is_file():
        ext = source_path.suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            # Single Image
            print(f"[RUN] Detecting on image: {source_path}")
            res = detector.predict_image(str(source_path), conf_threshold=args.conf, iou_threshold=args.iou)
            out_file = os.path.join(args.output_dir, f"annotated_{source_path.name}")
            cv2.imwrite(out_file, res["annotated_image"])
            print(f"[SUCCESS] Found {res['pothole_count']} pothole(s) in {res['latency_ms']}ms.")
            print(f"[SAVED] Result saved to: {out_file}")
            for d in res["detections"]:
                print(f"  - Pothole #{d['id']}: Severity={d['severity']}, Conf={d['confidence']:.2f}, Box={d['bbox']}")

        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            # Video File
            out_video = os.path.join(args.output_dir, f"annotated_{source_path.stem}.mp4")
            detector.predict_video(str(source_path), output_path=out_video, conf_threshold=args.conf, display=args.view)
        else:
            print(f"[ERROR] Unsupported file extension: {ext}")

    elif source_path.is_dir():
        # Batch Directory
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [f for f in source_path.iterdir() if f.suffix.lower() in valid_exts]
        print(f"[RUN] Processing directory of {len(image_files)} images...")
        total_found = 0
        for img_p in image_files:
            res = detector.predict_image(str(img_p), conf_threshold=args.conf)
            out_file = os.path.join(args.output_dir, f"annotated_{img_p.name}")
            cv2.imwrite(out_file, res["annotated_image"])
            total_found += res["pothole_count"]
            print(f"  Processed {img_p.name} -> {res['pothole_count']} pothole(s)")
        print(f"[SUCCESS] Batch completed. Total potholes detected: {total_found}. Output dir: {args.output_dir}")
    else:
        print(f"[ERROR] Source path does not exist: {args.source}")


if __name__ == "__main__":
    main()
