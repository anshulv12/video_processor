"""
GroundingDINO Object Detector - Optimized for Laundry Folding Tasks

Open-vocabulary detection that can identify clothing items, fabric, and laundry-related objects.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import torch
import warnings
import os
from pathlib import Path
import urllib.request
import groundingdino
from groundingdino.util.inference import Model as GroundingDINOModel
warnings.filterwarnings('ignore', category=FutureWarning)

# GroundingDINO weights URL
GROUNDING_DINO_WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
WEIGHTS_DIR = Path.home() / ".cache" / "groundingdino"
WEIGHTS_PATH = WEIGHTS_DIR / "groundingdino_swint_ogc.pth"

def download_weights():
    """Download GroundingDINO weights if not present."""
    if WEIGHTS_PATH.exists():
        print(f"✓ Weights found at {WEIGHTS_PATH}")
        return str(WEIGHTS_PATH)
    
    print(f"Downloading GroundingDINO weights (~700MB)...")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if count % 50 == 0:  # Update every 50 blocks
            print(f"  Progress: {percent}%", end='\r')
    
    urllib.request.urlretrieve(GROUNDING_DINO_WEIGHTS_URL, WEIGHTS_PATH, progress_hook)
    print(f"\n✓ Downloaded weights to {WEIGHTS_PATH}")
    return str(WEIGHTS_PATH)


class GroundingDINODetector:
    """
    Object detector using GroundingDINO for open-vocabulary detection.
    Optimized for egocentric laundry folding videos.
    """
    
    def __init__(
        self,
        box_threshold: float = 0.30,
        text_threshold: float = 0.20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GroundingDINO detector.
        
        Args:
            box_threshold: Confidence threshold for box detection (0.30 = more detections)
            text_threshold: Confidence threshold for text-image matching
            device: 'cuda' or 'cpu'
        """
        print(f"Initializing GroundingDINO detector on {device}...")
        
        # Download weights if needed
        weights_path = download_weights()
        
        # Get config path from installed package
        package_dir = Path(groundingdino.__file__).parent
        config_path = str(package_dir / "config" / "GroundingDINO_SwinT_OGC.py")
        
        # Load model
        self.model = GroundingDINOModel(
            model_config_path=config_path,
            model_checkpoint_path=weights_path,
            device=device
        )
        
        print(f"✓ GroundingDINO ready on {device}")
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        
        # Laundry-specific prompts
        self.laundry_prompts = [
            # Clothing items
            "shirt", "t-shirt", "blouse", "top",
            "pants", "trousers", "jeans", "shorts",
            "towel", "washcloth", "hand towel", "bath towel",
            "socks", "sock",
            "underwear", "undergarment",
            "jacket", "sweater", "hoodie",
            "dress", "skirt",
            "cloth", "fabric", "clothing",
            
            # Laundry tools & surfaces
            "basket", "laundry basket",
            "table", "surface", "folding table",
            "hanger", "clothes hanger",
            "detergent", "bottle",
            
            # Common household items that might appear
            "cup", "mug", "glass",
            "phone", "remote control",
            "book", "paper"
        ]
        
        # Create prompt string (format: "item1 . item2 . item3 .")
        self.prompt_string = " . ".join(self.laundry_prompts) + " ."
        
        print(f"✓ GroundingDINO ready with {len(self.laundry_prompts)} detection classes")
        print(f"  Detection prompts: {', '.join(self.laundry_prompts[:10])}...")
    
    def _is_in_relevant_region(self, bbox: List[float], frame_width: int, frame_height: int) -> bool:
        """
        Check if detection is in relevant region for egocentric video.
        Focus on central and lower parts where hands and objects typically appear.
        
        Args:
            bbox: [x1, y1, x2, y2] in normalized coordinates (0-1)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        
        Returns:
            True if object is in relevant region
        """
        x1, y1, x2, y2 = bbox
        
        # Convert to pixel coordinates for easier reasoning
        cx = (x1 + x2) / 2  # Center X (normalized)
        cy = (y1 + y2) / 2  # Center Y (normalized)
        
        # Egocentric region priorities:
        # - Horizontal: Center 80% of frame (exclude extreme edges)
        # - Vertical: Lower 70% of frame (where hands/objects typically are)
        
        horizontal_ok = 0.1 <= cx <= 0.9  # Center 80%
        vertical_ok = cy >= 0.3  # Lower 70% (skip top 30% usually just background)
        
        return horizontal_ok and vertical_ok
    
    def detect_objects_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect laundry and related objects in a single frame.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            List of detections with bbox, confidence, and label
        """
        # Convert BGR to RGB for GroundingDINO
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Run detection - returns (Detections object, List[str])
        detections, phrases = self.model.predict_with_caption(
            image=image_rgb,
            caption=self.prompt_string,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # Parse results
        results = []
        
        if detections is None or len(detections) == 0:
            return results
        
        # supervision.Detections object has: xyxy, confidence, class_id
        boxes = detections.xyxy  # numpy array [N, 4] in pixel coordinates
        confidences = detections.confidence  # numpy array [N]
        
        for i, (box, confidence, phrase) in enumerate(zip(boxes, confidences, phrases)):
            # Convert pixel coordinates to normalized [0-1]
            x1_px, y1_px, x2_px, y2_px = box
            x1 = x1_px / width
            y1 = y1_px / height
            x2 = x2_px / width
            y2 = y2_px / height
            
            # Filter by region (egocentric focus)
            if not self._is_in_relevant_region([x1, y1, x2, y2], width, height):
                continue
            
            # Clean up phrase (remove extra spaces)
            phrase = phrase.strip()
            
            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],  # Normalized [0-1]
                "confidence": float(confidence),
                "label": phrase,
                "class_id": -1,  # Open vocabulary, no fixed class ID
                "detector": "GroundingDINO"
            })
        
        return results
    
    def detect_objects_video(
        self, 
        video_path: str, 
        sample_rate: int = 1,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in video frames.
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (1 = every frame, 5 = every 5th frame)
            progress_callback: Optional callback function(frame_index, total_frames)
        
        Returns:
            List of frame detections: [{"frame_index": int, "detections": [...]}, ...]
        """
        video_cap = cv2.VideoCapture(video_path)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detections = []
        frame_index = 0
        
        print(f"Processing video: {total_frames} frames (sampling every {sample_rate} frame(s))")
        
        while video_cap.isOpened():
            success, frame = video_cap.read()
            if not success:
                break
            
            # Skip if frame is None or empty
            if frame is None or frame.size == 0:
                frame_index += 1
                continue
            
            # Sample frames
            if frame_index % sample_rate == 0:
                try:
                    frame_detections = self.detect_objects_frame(frame)
                except Exception as e:
                    print(f"Warning: Failed to detect objects in frame {frame_index}: {e}")
                    frame_detections = []
                
                detections.append({
                    "frame_index": frame_index,
                    "detections": frame_detections
                })
                
                if progress_callback:
                    progress_callback(frame_index, total_frames)
                
                if frame_index % 100 == 0:
                    print(f"  Processed frame {frame_index}/{total_frames}")
            
            frame_index += 1
        
        video_cap.release()
        print(f"✓ Detection complete: {len(detections)} frames processed")
        
        return detections
    
    def visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame for visualization.
        
        Args:
            frame: BGR image
            detections: List of detections from detect_objects_frame
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
        for det in detections:
            # Convert normalized bbox to pixel coordinates
            x1, y1, x2, y2 = det['bbox']
            x1_px = int(x1 * width)
            y1_px = int(y1 * height)
            x2_px = int(x2 * width)
            y2_px = int(y2 * height)
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated, (x1_px, y1_px), (x2_px, y2_px), color, 2)
            
            # Draw label
            label = f"{det['label']}: {det['confidence']:.2f}"
            cv2.putText(
                annotated, 
                label, 
                (x1_px, y1_px - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
        
        return annotated


if __name__ == "__main__":
    # Test the detector
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python grounding_dino_detector.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Initialize detector
    detector = GroundingDINODetector()
    
    # Test on first frame
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    
    if success:
        print("\nDetecting objects in first frame...")
        detections = detector.detect_objects_frame(frame)
        
        print(f"\nFound {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['label']}: {det['confidence']:.3f} at {det['bbox']}")
        
        # Visualize
        annotated = detector.visualize_detections(frame, detections)
        cv2.imwrite("test_detection.jpg", annotated)
        print("\n✓ Saved visualization to test_detection.jpg")
    else:
        print("Failed to read video")

