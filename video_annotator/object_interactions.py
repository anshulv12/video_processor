from typing import List, Dict, Any
import numpy as np

class ObjectInteractions:
    def __init__(self, interaction_threshold: float = 0.5, assume_clothing: bool = True):
        """
        Args:
            interaction_threshold: Distance threshold for hand-object interaction
        """
        self.interaction_threshold = interaction_threshold
        self.assume_clothing = assume_clothing

    def is_hand_interacting_with_object(self, pose_hand: Dict[str, Any], object_frame: Dict[str, Any]) -> List[str]:
        # For laundry folding: assume hands are always manipulating clothing
        if self.assume_clothing:
            return ["clothing"]  # Hardcoded assumption for laundry tasks
        
        # Original logic (for when YOLO can actually detect relevant objects)
        max_points = [float('-inf'), float('-inf')]
        min_points = [float('inf'), float('inf')]
        for _, joint in pose_hand["joints"].items():
            x, y, z = joint["x"], joint["y"], joint["z"]
            max_points[0], max_points[1] = max(max_points[0], x), max(max_points[1], y)
            min_points[0], min_points[1] = min(min_points[0], x), min(min_points[1], y)
        interactions = []
        for object in object_frame["detections"]:
            object_bbox = object["bbox"]
            dist = np.linalg.norm([object_bbox[0] - min_points[0], object_bbox[1] - min_points[1]])
            is_crossing = min_points[0] <= object_bbox[0] <= max_points[0] and min_points[1] <= object_bbox[1] <= max_points[1]
            if dist <= self.interaction_threshold or is_crossing:
                interactions.append(object["label"])
        return interactions

    def compute_interaction_frame(self, pose_frame: Dict[str, Any], object_frame: Dict[str, Any]) -> Dict[str, Any]:
        interactions = {"frame_index": pose_frame["frame_index"]}
        for pose_hand in pose_frame["hands"]:
            object_interactions = self.is_hand_interacting_with_object(pose_hand, object_frame)
            interactions[pose_hand["handedness"]] = {
                "objects": object_interactions
            }
        return interactions

    def compute_interaction_video(self, pose_frames: List[Dict[str, Any]], object_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        interactions = []
        num_frames = len(pose_frames)
        for i in range(num_frames):
            pose_frame = pose_frames[i]
            object_frame = object_frames[i]
            interaction_frame = self.compute_interaction_frame(pose_frame, object_frame)
            interactions.append(interaction_frame)
        return interactions