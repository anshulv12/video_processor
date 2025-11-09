import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import cv2

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.6):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def detect_objects_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        if len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # normalize bbox to 0-1
                width, height = frame.shape[1], frame.shape[0]
                normalized_x1, normalized_y1, normalized_x2, normalized_y2 = x1 / width, y1 / height, x2 / width, y2 / height
                confidence = box.conf.item()
                class_id = box.cls.item()
                label = self.model.names[class_id]
                detections.append({
                    "bbox": [normalized_x1, normalized_y1, normalized_x2, normalized_y2],
                    "confidence": confidence,
                    "label": label
                })
        return detections
    
    def detect_objects_video(self, video_path: str) -> List[Dict[str, Any]]:
        video_cap = cv2.VideoCapture(video_path)
        detections = []
        frame_index = 0
        while video_cap.isOpened():
            success, frame = video_cap.read()
            if not success:
                break
            detections.append({"frame_index": frame_index, "detections": self.detect_objects_frame(frame)})
            frame_index += 1
        video_cap.release()
        return detections

if __name__ == "__main__":
    detector = ObjectDetector()
    detections = detector.detect_objects_video("video_data/0.mp4")
    print(detections)