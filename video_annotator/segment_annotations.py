from typing import List, Dict, Any, Optional


class SegmentAnnotater:
    def __init__(self):
        pass
    
    def annotate_segments(
        self,
        text_annotations: List[Dict[str, Any]],
        pose_annotations: List[Dict[str, Any]],
        pose_flows: List[Dict[str, Any]],
        object_annotations: List[Dict[str, Any]],
        object_interactions: List[Dict[str, Any]],
        frame_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        fps: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Group annotations, poses, flows, objects, and interactions by segment."""
        frame_metadata = frame_metadata or {}

        pose_map = {frame["frame_index"]: frame for frame in pose_annotations}
        flow_map = {frame["frame_index"]: frame for frame in pose_flows}
        object_map = {frame["frame_index"]: frame for frame in object_annotations}
        interaction_map = {frame["frame_index"]: frame for frame in object_interactions}

        segments: List[Dict[str, Any]] = []
        for annotation in text_annotations:
            frame_start = annotation["frame_start"]
            frame_end = annotation["frame_end"]
            segment_index = annotation.get("segment_index", len(segments))

            timestamp_start = annotation.get("timestamp_start")
            timestamp_end = annotation.get("timestamp_end")
            if timestamp_start is None and fps:
                timestamp_start = frame_start / fps
            if timestamp_end is None and fps:
                timestamp_end = frame_end / fps

            cur_segment = {
                "segment_index": segment_index,
                "text_description": annotation.get("text_description"),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "key_frame_path": annotation.get("key_frame_path"),
                "frames": [],
            }

            for frame_index in range(frame_start, frame_end + 1):
                frame_meta = frame_metadata.get(frame_index, {})
                timestamp = frame_meta.get("timestamp")
                if timestamp is None and fps:
                    timestamp = frame_index / fps

                cur_frame: Dict[str, Any] = {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                }

                if "frame_path" in frame_meta:
                    cur_frame["frame_path"] = frame_meta["frame_path"]
                if "frame_filename" in frame_meta:
                    cur_frame["frame_filename"] = frame_meta["frame_filename"]
                if "frame_url" in frame_meta:
                    cur_frame["frame_url"] = frame_meta["frame_url"]

                if frame_index in pose_map:
                    cur_frame["poses"] = pose_map[frame_index]
                if frame_index in flow_map:
                    cur_frame["pose_flows"] = flow_map[frame_index]
                if frame_index in object_map:
                    cur_frame["objects"] = object_map[frame_index]
                if frame_index in interaction_map:
                    cur_frame["object_interactions"] = interaction_map[frame_index]

                cur_segment["frames"].append(cur_frame)

            segments.append(cur_segment)

        return segments