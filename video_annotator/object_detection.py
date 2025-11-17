import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Set
import cv2

class ObjectDetector:
    # Classes to exclude in egocentric video (irrelevant for hand-object tasks)
    EGOCENTRIC_EXCLUDE_CLASSES = {
        'person',      # Camera wearer or other people
        'traffic light', 'stop sign', 'parking meter',  # Outdoor/traffic (unlikely in egocentric tasks)
        'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',  # Vehicles (background)
        'fire hydrant', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',  # Animals
        'bear', 'zebra', 'giraffe',
        'couch', 'bed', 'dining table', 'tv', 'bench',  # Large furniture (usually background, not manipulated)
    }
    
    # Objects commonly manipulated in egocentric tasks (lower confidence threshold for these)
    MANIPULABLE_OBJECTS = {
        'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'potted plant', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'wine glass', 'tie', 'backpack', 'umbrella', 'handbag', 'suitcase'
    }
    
    def __init__(
        self, 
        model_name: str = "yolov8m.pt",  # Changed to medium model (better accuracy)
        confidence_threshold: float = 0.5,  # Lowered for better recall
        exclude_classes: Set[str] = None,
        focus_on_center: bool = True  # Focus on center region where hands typically are
    ):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.focus_on_center = focus_on_center
        
        # Combine default and custom exclude classes
        self.exclude_classes = self.EGOCENTRIC_EXCLUDE_CLASSES.copy()
        if exclude_classes:
            self.exclude_classes.update(exclude_classes)
        
        print(f"ObjectDetector initialized with {model_name}")
        print(f"Excluding {len(self.exclude_classes)} classes: {sorted(list(self.exclude_classes)[:5])}...")

    def _is_in_relevant_region(self, bbox: List[float], frame_shape: tuple) -> bool:
        """
        Check if object is in the relevant region for egocentric video.
        In first-person view, hands and manipulated objects are typically in:
        - Lower 60% of frame (hands naturally work in lower field of view)
        - Central 80% horizontally (edges are usually background)
        """
        if not self.focus_on_center:
            return True
        
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # Calculate bbox center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalize to 0-1
        norm_x = center_x / width
        norm_y = center_y / height
        
        # Check if in relevant region:
        # - Horizontally: between 10% and 90% (central 80%)
        # - Vertically: between 20% and 100% (lower 80%, hands rarely at very top)
        horizontal_ok = 0.1 <= norm_x <= 0.9
        vertical_ok = 0.2 <= norm_y <= 1.0
        
        return horizontal_ok and vertical_ok

    def detect_objects_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in a frame, filtering irrelevant classes."""
        # Run with lower confidence to catch hard-to-detect items (like clothing)
        results = self.model(frame, conf=0.25, verbose=False)  # Lower threshold initially
        
        detections = []
        if len(results) > 0:
            for box in results[0].boxes:    
                class_id = int(box.cls.item())
                label = self.model.names[class_id]
                confidence = box.conf.item()
                
                # Skip excluded classes (person, animals, vehicles, large furniture, etc.)
                if label in self.exclude_classes:
                    continue
                
                # Apply different thresholds based on object type
                if label in self.MANIPULABLE_OBJECTS:
                    # More lenient for commonly manipulated objects
                    min_conf = 0.3
                else:
                    # Stricter for other objects
                    min_conf = self.confidence_threshold
                
                if confidence < min_conf:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Skip objects not in relevant region
                if not self._is_in_relevant_region([x1, y1, x2, y2], frame.shape):
                    continue
                
                # Normalize bbox to 0-1
                height, width = frame.shape[:2]
                normalized_bbox = [
                    x1 / width, 
                    y1 / height, 
                    x2 / width, 
                    y2 / height
                ]
                
                detections.append({
                    "bbox": normalized_bbox,
                    "confidence": confidence,
                    "label": label,
                    "class_id": class_id
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
