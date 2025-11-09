import json
import numpy as np
from typing import List, Dict, Any

class PoseFlowCalculator:
    def __init__(self):
        pass

    def compute_joint_flow(
        self, prev_frame: Dict[str, Any], cur_frame: Dict[str, Any], fps: int
    ) -> Dict[str, Any]:
        flows = {"frame_index": cur_frame["frame_index"], "hands": []}
        for hand_idx, cur_hand in enumerate(cur_frame["hands"]):
            hand_flow = {"handedness": cur_hand["handedness"], "joints": {}}
            if hand_idx >= len(prev_frame["hands"]):
                continue
            prev_hand = prev_frame["hands"][hand_idx]
            for joint_name, cur_joint in cur_hand["joints"].items():
                if joint_name not in prev_hand["joints"]:
                    continue

                prev_joint = prev_hand["joints"][joint_name]

                x1, y1, z1 = prev_joint["x"], prev_joint["y"], prev_joint["z"]
                x2, y2, z2 = cur_joint["x"], cur_joint["y"], cur_joint["z"]

                dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)

                speed = np.sqrt(dx**2 + dy**2 + dz**2) * fps
                hand_flow["joints"][joint_name] = speed

            flows["hands"].append(hand_flow)

        return flows

    def compute_all_flows(self, pose_frames: List[Dict[str, Any]], fps: int) -> List[Dict[str, Any]]:
        flows = []
        for i in range(1, len(pose_frames)):
            prev_frame = pose_frames[i - 1]
            cur_frame = pose_frames[i]
            frame_flow = self.compute_joint_flow(prev_frame, cur_frame, fps)
            flows.append(frame_flow)
        return flows