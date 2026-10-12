"""
track.py - Multi-Object Tracking & Unique Pothole Counter for Video Streams
Utilizes ByteTrack / BoT-SORT to assign persistent IDs across frames,
preventing redundant counts of the same pothole as a vehicle drives past.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from detect import DEFAULT_MODEL_PATH, classify_severity, SEVERITY_COLORS


class PotholeTracker:
    """Multi-Object Tracking and Unique Pothole Aggregator."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, tracker_type: str = "bytetrack.yaml"):
        if not os.path.exists(model_path):
            fallback = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Notebook", "runs", "detect", "pothole_yolov8_baseline-2", "weights", "best.pt"
            )
            if os.path.exists(fallback):
                model_path = fallback
            else:
                raise FileNotFoundError(f"Weights not found: {model_path}")

        self.model = YOLO(model_path)
        self.tracker_type = tracker_type
        print(f"[TRACKER] Initialized YOLOv8 with Tracker: {tracker_type}")

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = "outputs/tracked_potholes.mp4",
        conf_threshold: float = 0.25,
        display: bool = False
    ) -> Dict:
        """
        Processes video frames with object tracking.
        Maintains unique track IDs to prevent duplicate detections.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area = width * height

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        unique_potholes: Dict[int, Dict] = {}
        frame_idx = 0
        t0 = time.time()

        print(f"[RUN] Tracking potholes on video: {video_path} ({total_frames} frames)...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                source=frame,
                conf=conf_threshold,
                tracker=self.tracker_type,
                persist=True,
                verbose=False
            )[0]

            annotated_frame = frame.copy()

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    box_w = max(0, x2 - x1)
                    box_h = max(0, y2 - y1)
                    area_ratio = (box_w * box_h) / max(1, frame_area)
                    severity = classify_severity(area_ratio)

                    if track_id not in unique_potholes:
                        unique_potholes[track_id] = {
                            "id": track_id,
                            "first_frame": frame_idx,
                            "max_conf": float(conf),
                            "severity": severity,
                            "area_ratio": area_ratio
                        }
                    else:
                        if conf > unique_potholes[track_id]["max_conf"]:
                            unique_potholes[track_id]["max_conf"] = float(conf)
                            unique_potholes[track_id]["severity"] = severity
                            unique_potholes[track_id]["area_ratio"] = area_ratio

                    color = SEVERITY_COLORS.get(severity, (0, 0, 255))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID #{track_id} | {severity} ({conf:.2f})"
                    cv2.putText(
                        annotated_frame, label, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
                    )

            # Heads Up Display (HUD)
            elapsed = max(0.001, time.time() - t0)
            cur_fps = (frame_idx + 1) / elapsed
            hud_bg = np.zeros((70, 380, 3), dtype=np.uint8)
            overlay = annotated_frame[10:80, 10:390]
            if overlay.shape == hud_bg.shape:
                cv2.addWeighted(overlay, 0.3, hud_bg, 0.7, 0, annotated_frame[10:80, 10:390])

            cv2.putText(
                annotated_frame, f"Unique Potholes Encountered: {len(unique_potholes)}",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
            )
            cv2.putText(
                annotated_frame, f"Frame: {frame_idx}/{total_frames} | Live FPS: {cur_fps:.1f}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )

            if writer:
                writer.write(annotated_frame)

            if display:
                cv2.imshow("Pothole Tracking & Count", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        total_time = time.time() - t0
        avg_fps = frame_idx / max(0.001, total_time)

        print(f"[COMPLETE] Processed {frame_idx} frames in {total_time:.2f}s ({avg_fps:.1f} FPS).")
        print(f"[RESULT] Total unique potholes encountered: {len(unique_potholes)}")

        return {
            "total_frames": frame_idx,
            "unique_potholes_count": len(unique_potholes),
            "pothole_summary": list(unique_potholes.values()),
            "average_fps": round(avg_fps, 2),
            "output_video": output_path
        }


def main():
    parser = argparse.ArgumentParser(description="Pothole Video Tracking & Unique Counter")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--output", type=str, default="outputs/tracked_potholes.mp4", help="Output path.")
    parser.add_argument("--view", action="store_true", help="Display window while processing.")
    args = parser.parse_args()

    tracker = PotholeTracker(model_path=args.model)
    tracker.process_video(args.video, output_path=args.output, conf_threshold=args.conf, display=args.view)


if __name__ == "__main__":
    main()
