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
        """
        Group annotations, poses, flows, objects, and interactions by segment.
        Ensures EVERY frame in the video is covered (no gaps).
        """
        frame_metadata = frame_metadata or {}

        pose_map = {frame["frame_index"]: frame for frame in pose_annotations}
        flow_map = {frame["frame_index"]: frame for frame in pose_flows}
        object_map = {frame["frame_index"]: frame for frame in object_annotations}
        interaction_map = {frame["frame_index"]: frame for frame in object_interactions}

        # Determine the total range of frames in the video
        max_frame = 0
        for data_map in [pose_map, flow_map, object_map, interaction_map]:
            if data_map:
                max_frame = max(max_frame, max(data_map.keys()))
        
        # Also check text annotations
        for annotation in text_annotations:
            max_frame = max(max_frame, annotation["frame_end"])
        
        print(f"Total video frames: 0 to {max_frame}")

        segments: List[Dict[str, Any]] = []
        
        # Process text annotations and fill in ALL frames between segments
        for seg_idx, annotation in enumerate(text_annotations):
            frame_start = annotation["frame_start"]
            frame_end = annotation["frame_end"]
            
            # If this is not the first segment, extend the previous segment
            # to cover frames up to (but not including) this segment's start
            if seg_idx > 0 and segments:
                prev_segment = segments[-1]
                prev_end = prev_segment["frame_end"]
                
                # Fill gap between previous segment and this one
                if frame_start > prev_end + 1:
                    # Extend previous segment to cover the gap
                    for frame_index in range(prev_end + 1, frame_start):
                        frame_data = self._create_frame_data(
                            frame_index, fps, frame_metadata,
                            pose_map, flow_map, object_map, interaction_map
                        )
                        prev_segment["frames"].append(frame_data)
                    prev_segment["frame_end"] = frame_start - 1
                    prev_segment["timestamp_end"] = (frame_start - 1) / fps

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

            # Add all frames in this segment
            for frame_index in range(frame_start, frame_end + 1):
                frame_data = self._create_frame_data(
                    frame_index, fps, frame_metadata,
                    pose_map, flow_map, object_map, interaction_map
                )
                cur_segment["frames"].append(frame_data)

            segments.append(cur_segment)
        
        # Extend the last segment to cover any remaining frames
        if segments and max_frame > segments[-1]["frame_end"]:
            last_segment = segments[-1]
            for frame_index in range(last_segment["frame_end"] + 1, max_frame + 1):
                frame_data = self._create_frame_data(
                    frame_index, fps, frame_metadata,
                    pose_map, flow_map, object_map, interaction_map
                )
                last_segment["frames"].append(frame_data)
            last_segment["frame_end"] = max_frame
            last_segment["timestamp_end"] = max_frame / fps
        
        print(f"Created {len(segments)} segments covering all {max_frame + 1} frames")

        return segments
    
    def _create_frame_data(
        self,
        frame_index: int,
        fps: float,
        frame_metadata: Dict[int, Dict[str, Any]],
        pose_map: Dict[int, Dict[str, Any]],
        flow_map: Dict[int, Dict[str, Any]],
        object_map: Dict[int, Dict[str, Any]],
        interaction_map: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create frame data dict with all available information."""
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
            pose_data = pose_map[frame_index].copy()
            pose_data.pop("frame_index", None)
            cur_frame["poses"] = pose_data
            
        if frame_index in flow_map:
            flow_data = flow_map[frame_index].copy()
            flow_data.pop("frame_index", None)
            cur_frame["pose_flows"] = flow_data
            
        if frame_index in object_map:
            object_data = object_map[frame_index].copy()
            object_data.pop("frame_index", None)
            cur_frame["objects"] = object_data
            
        if frame_index in interaction_map:
            interaction_data = interaction_map[frame_index].copy()
            interaction_data.pop("frame_index", None)
            cur_frame["object_interactions"] = interaction_data

        return cur_frame
