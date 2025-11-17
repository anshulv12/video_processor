from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

class RobustFrameAnnotator:
    """
    Frame annotator with robust scene change detection using multi-factor analysis.
    Prevents false boundaries from temporary variations like motion blur or lighting changes.
    """
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32",
        base_similarity_threshold: float = 0.85,
        strict_similarity_threshold: float = 0.90,
        min_segment_length: int = 15,
        lookahead_window: int = 3,
        idle_velocity_threshold: float = 1.0,  # Max velocity for idle detection (joint speed)
        remove_idle_segments: bool = True  # Filter out idle segments from output
    ):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.base_threshold = base_similarity_threshold
        self.strict_threshold = strict_similarity_threshold
        self.min_segment_length = min_segment_length
        self.lookahead_window = lookahead_window
        self.idle_velocity_threshold = idle_velocity_threshold
        self.remove_idle_segments = remove_idle_segments
        self.model.eval()
        self.client = genai.Client()

    def embed_frame(self, frame_path: str) -> np.ndarray:
        """Extract CLIP embedding from a frame."""
        image = Image.open(frame_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        embedding = outputs[0].cpu().numpy()
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def compute_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding_a, embedding_b))
    
    def should_create_boundary(
        self, 
        current_idx: int,
        all_embeddings: List[np.ndarray],
        segment_frames: List[int],
        frame_metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Multi-factor decision: should we create a segment boundary?
        
        Factors considered:
        1. Similarity with recent frames in current segment
        2. Look-ahead: are future frames internally consistent (new stable segment)?
        3. Gradient analysis: gradual vs sharp change
        4. Minimum segment length enforcement
        """
        
        current_frame_idx = frame_metadata[current_idx]["frame_index"]
        current_embedding = all_embeddings[current_idx]
        
        # Factor 0: Enforce minimum segment length
        if len(segment_frames) < self.min_segment_length:
            return False
        
        # Factor 1: Check similarity with last 3 frames in segment
        recent_indices = segment_frames[-3:] if len(segment_frames) >= 3 else segment_frames
        recent_similarities = [
            self.compute_similarity(current_embedding, all_embeddings[idx])
            for idx in recent_indices
        ]
        avg_recent_sim = np.mean(recent_similarities)
        
        # If very similar to recent frames, definitely don't split
        if avg_recent_sim >= self.strict_threshold:
            return False
        
        # If very dissimilar, check other factors
        if avg_recent_sim < self.base_threshold:
            # Factor 2: Look-ahead - are FUTURE frames internally consistent?
            # If current frame starts a new stable segment, future frames should be similar to current
            lookahead_indices = list(range(
                current_idx + 1, 
                min(current_idx + 1 + self.lookahead_window, len(all_embeddings))
            ))
            
            if lookahead_indices:
                # Check if future frames are similar to CURRENT frame (internal consistency of new segment)
                future_to_current_similarities = [
                    self.compute_similarity(current_embedding, all_embeddings[idx])
                    for idx in lookahead_indices
                ]
                
                # If future frames are NOT consistent with current frame, might be noise
                if np.mean(future_to_current_similarities) < self.base_threshold:
                    print(f"Frame {current_frame_idx}: Future frames inconsistent, might be noise")
                    return False
                
                # Check internal consistency of future frames with each other
                if len(lookahead_indices) >= 2:
                    future_internal_sims = []
                    for i in range(len(lookahead_indices) - 1):
                        sim = self.compute_similarity(
                            all_embeddings[lookahead_indices[i]], 
                            all_embeddings[lookahead_indices[i + 1]]
                        )
                        future_internal_sims.append(sim)
                    
                    # If future frames are consistent with each other, this is a stable new segment
                    if np.mean(future_internal_sims) >= self.strict_threshold:
                        print(f"Frame {current_frame_idx}: Future frames form consistent new segment (sim={avg_recent_sim:.3f})")
                        return True
            
            # Factor 3: Gradient analysis - is this a sharp or gradual change?
            if len(segment_frames) >= 3:
                similarity_trend = [
                    self.compute_similarity(
                        all_embeddings[segment_frames[-3]], 
                        all_embeddings[segment_frames[-2]]
                    ),
                    self.compute_similarity(
                        all_embeddings[segment_frames[-2]], 
                        all_embeddings[segment_frames[-1]]
                    ),
                    avg_recent_sim
                ]
                
                # Calculate gradient (rate of change)
                gradients = np.diff(similarity_trend)
                
                # If gradual decline (small negative gradients), likely camera motion
                if all(g > -0.08 for g in gradients):
                    print(f"Frame {current_frame_idx}: Gradual change detected, keeping in segment")
                    return False
            
            # Dissimilar to previous segment
            print(f"Frame {current_frame_idx}: Scene change detected (sim={avg_recent_sim:.3f})")
            return True
        
        # In between thresholds - use strict threshold for consistency
        return avg_recent_sim < self.strict_threshold
    
    def refine_boundary(
        self,
        coarse_boundary_frame: int,
        previous_segment_frame: int,
        frame_metadata: List[Dict[str, Any]]
    ) -> int:
        """
        Refine the boundary by checking frames between the last sampled frame
        and the detected boundary frame to find the exact moment of change.
        
        Since we only extract frames at intervals (e.g., every 30 frames),
        we check if there are any extracted frames in the gap. If yes, we find
        the best boundary among them. If no, we accept the coarse boundary.
        
        Args:
            coarse_boundary_frame: Metadata index where we detected the change (sampled)
            previous_segment_frame: Metadata index of last frame in previous segment (sampled)
            frame_metadata: All frame metadata from frame extractor
        
        Returns:
            Refined metadata index where actual change occurs
        """
        # Find actual video frame numbers
        prev_actual_frame = frame_metadata[previous_segment_frame]["frame_index"]
        boundary_actual_frame = frame_metadata[coarse_boundary_frame]["frame_index"]
        
        gap_size = boundary_actual_frame - prev_actual_frame
        
        # If they're adjacent in actual video frames or extraction indices, no refinement needed
        if gap_size <= 1 or coarse_boundary_frame - previous_segment_frame <= 1:
            return coarse_boundary_frame
        
        print(f"Refining boundary between extracted frames (actual video frames {prev_actual_frame} and {boundary_actual_frame}, gap={gap_size})...")
        
        # Find all extracted frames in the gap between prev and boundary
        gap_meta_indices = []
        for meta_idx in range(previous_segment_frame + 1, coarse_boundary_frame):
            gap_meta_indices.append(meta_idx)
        
        if not gap_meta_indices:
            # No extracted frames in gap, accept coarse boundary
            print(f"  No extracted frames in gap, using coarse boundary at frame {boundary_actual_frame}")
            return coarse_boundary_frame
        
        print(f"  Found {len(gap_meta_indices)} extracted frames in gap, checking each...")
        
        # Get reference embeddings (reuse if already computed)
        prev_path = frame_metadata[previous_segment_frame]["frame_path"]
        boundary_path = frame_metadata[coarse_boundary_frame]["frame_path"]
        prev_embedding = self.embed_frame(prev_path)
        boundary_embedding = self.embed_frame(boundary_path)
        
        # Find the best boundary point in the gap
        best_boundary_idx = coarse_boundary_frame
        best_score = -1
        
        for meta_idx in gap_meta_indices:
            gap_path = frame_metadata[meta_idx]["frame_path"]
            gap_emb = self.embed_frame(gap_path)
            gap_actual_frame = frame_metadata[meta_idx]["frame_index"]
            
            # Transition score: want low similarity to prev + high similarity to boundary
            sim_to_prev = self.compute_similarity(gap_emb, prev_embedding)
            sim_to_boundary = self.compute_similarity(gap_emb, boundary_embedding)
            
            # Score: how well does this frame separate the two segments?
            transition_score = sim_to_boundary - sim_to_prev
            
            if transition_score > best_score:
                best_score = transition_score
                best_boundary_idx = meta_idx
        
        actual_refined_frame = frame_metadata[best_boundary_idx]["frame_index"]
        print(f"  Refined boundary to actual video frame {actual_refined_frame} (score={best_score:.3f})")
        
        return best_boundary_idx
    
    def detect_idle_segment(
        self,
        segment_frame_indices: List[int],
        frame_metadata: List[Dict[str, Any]],
        pose_flows: Dict[int, Dict[str, Any]],
        object_interactions: Dict[int, Dict[str, Any]],
        pose_estimates: Dict[int, Dict[str, Any]],
        all_embeddings: List[np.ndarray]
    ) -> tuple[bool, float, bool, bool, float]:
        """
        Detect if a segment is idle (no meaningful activity).
        
        Idle conditions:
        1. No hands visible (camera movement, looking around) → idle
        2. Hands visible but low velocity AND no object interactions → idle
        3. Low internal similarity (incoherent frames, random movement) → idle
        
        Pose flow structure: {"frame_index": 123, "hands": [{"handedness": "Left", "joints": {"wrist": 0.05, ...}}]}
        
        Returns:
            (is_idle, avg_velocity, has_interactions, hands_visible, internal_similarity)
        """
        velocities = []
        has_any_interaction = False
        frames_with_hands = 0
        total_frames_checked = 0
        
        for idx in segment_frame_indices:
            frame_idx = frame_metadata[idx]["frame_index"]
            total_frames_checked += 1
            
            # Check if hands are visible in pose estimates
            if frame_idx in pose_estimates:
                pose = pose_estimates[frame_idx]
                if 'hands' in pose and isinstance(pose['hands'], list) and len(pose['hands']) > 0:
                    frames_with_hands += 1
            
            # Check pose flow velocity
            if frame_idx in pose_flows:
                flow = pose_flows[frame_idx]
                
                # Parse actual pose flow structure
                if 'hands' in flow and isinstance(flow['hands'], list):
                    for hand in flow['hands']:
                        if 'joints' in hand and isinstance(hand['joints'], dict):
                            # Get all joint speeds and average them
                            joint_speeds = [speed for speed in hand['joints'].values() if isinstance(speed, (int, float))]
                            if joint_speeds:
                                hand_avg_speed = np.mean(joint_speeds)
                                velocities.append(hand_avg_speed)
            
            # Check object interactions
            # Structure: {"frame_index": 123, "Left": {"objects": ["cup"]}, "Right": {"objects": []}}
            if frame_idx in object_interactions:
                interactions = object_interactions[frame_idx]
                if isinstance(interactions, dict):
                    # Check if any hand has interactions
                    for hand_key in ['Left', 'Right']:
                        if hand_key in interactions:
                            hand_data = interactions[hand_key]
                            if isinstance(hand_data, dict):
                                objects = hand_data.get('objects', [])
                                if objects and len(objects) > 0:
                                    has_any_interaction = True
                                    break
        
        # Calculate hand visibility percentage
        hands_visible_percent = frames_with_hands / total_frames_checked if total_frames_checked > 0 else 0
        hands_visible = hands_visible_percent > 0.3  # At least 30% of frames have hands
        
        avg_velocity = np.mean(velocities) if velocities else 0
        
        # Check internal similarity (are frames coherent?)
        internal_similarities = []
        if len(segment_frame_indices) >= 2:
            # Sample pairs of frames to check coherence
            sample_size = min(10, len(segment_frame_indices) // 2)
            for i in range(sample_size):
                idx1 = segment_frame_indices[i * 2]
                idx2 = segment_frame_indices[min(i * 2 + 1, len(segment_frame_indices) - 1)]
                sim = self.compute_similarity(all_embeddings[idx1], all_embeddings[idx2])
                internal_similarities.append(sim)
        
        internal_similarity = np.mean(internal_similarities) if internal_similarities else 1.0
        low_coherence = internal_similarity < 0.75  # Frames don't form coherent sequence
        
        # Idle if:
        # 1. Hands not visible (camera movement, looking away) → IDLE
        # 2. Low internal similarity (incoherent frames) → IDLE  
        # 3. Hands visible but low velocity AND no interactions → IDLE
        is_idle = (
            (not hands_visible) or 
            low_coherence or 
            (avg_velocity < self.idle_velocity_threshold and not has_any_interaction)
        )
        
        # Debug logging
        frame_start = frame_metadata[segment_frame_indices[0]]["frame_index"]
        frame_end = frame_metadata[segment_frame_indices[-1]]["frame_index"]
        
        if not hands_visible:
            print(f"    Segment (frames {frame_start}-{frame_end}): Hands visible in only {frames_with_hands}/{total_frames_checked} frames → IDLE (camera movement)")
        elif low_coherence:
            print(f"    Segment (frames {frame_start}-{frame_end}): Low internal similarity ({internal_similarity:.3f}) → IDLE (incoherent)")
        elif len(velocities) == 0:
            print(f"    WARNING: No velocity data found for frames {frame_start}-{frame_end}")
        
        return is_idle, float(avg_velocity), has_any_interaction, hands_visible, float(internal_similarity)
    
    def detect_repetitive_instances(
        self,
        segment_frame_indices: List[int],
        all_embeddings: List[np.ndarray],
        frame_metadata: List[Dict[str, Any]],
        min_instance_length: int = 20
    ) -> List[List[int]]:
        """
        Detect repetitive instances within a segment by finding state resets.
        
        A state reset = visual state returns to start (e.g., picked up new item).
        
        Returns:
            List of instance frame index lists (split points)
        """
        if len(segment_frame_indices) < min_instance_length * 2:
            # Too short to contain multiple instances
            return [segment_frame_indices]
        
        # Get start embedding (initial state)
        start_embedding = all_embeddings[segment_frame_indices[0]]
        
        # Track potential reset points
        reset_points = []
        last_reset = 0
        
        # Skip first 10 frames (let first instance start)
        for i in range(10, len(segment_frame_indices)):
            idx = segment_frame_indices[i]
            current_embedding = all_embeddings[idx]
            
            # Check if returned to start state
            sim_to_start = self.compute_similarity(current_embedding, start_embedding)
            
            # Check dissimilarity to recent frames
            if i >= 5:
                prev_idx = segment_frame_indices[i - 5]
                prev_embedding = all_embeddings[prev_idx]
                sim_to_prev = self.compute_similarity(current_embedding, prev_embedding)
            else:
                sim_to_prev = 1.0
            
            # State reset detected: returned to start after being different
            if sim_to_start > 0.88 and sim_to_prev < 0.80:
                # Ensure minimum instance length
                if i - last_reset >= min_instance_length:
                    reset_points.append(i)
                    last_reset = i
                    frame_idx = frame_metadata[idx]["frame_index"]
                    print(f"  Repetition detected at frame {frame_idx} (similarity to start: {sim_to_start:.3f})")
        
        # Split into instances
        if not reset_points:
            return [segment_frame_indices]
        
        instances = []
        start_i = 0
        for reset_i in reset_points:
            instances.append(segment_frame_indices[start_i:reset_i])
            start_i = reset_i
        
        # Add final instance
        instances.append(segment_frame_indices[start_i:])
        
        # Filter out instances that are too short
        instances = [inst for inst in instances if len(inst) >= min_instance_length]
        
        return instances if instances else [segment_frame_indices]
    
    def generate_text_from_image(self, image_path: str) -> str:
        """Generate text description using Gemini."""
        image = Image.open(image_path)
        prompt = """This is a representative frame from a VIDEO SEGMENT (multiple frames) in a first-person (egocentric) video where the camera is mounted on the person's head.

Your task: Describe the ACTION or TASK the person is performing that transforms the state of objects from the START to the END of this video segment. Use 5-8 words maximum.

IMPORTANT:
- Describe the TRANSFORMATIVE ACTION, not just what is being held
- Think: "What is the person DOING to change the state of objects?"
- Use verbs that imply change or transformation (folding, opening, closing, pouring, cutting, assembling, etc.)
- Avoid static verbs like "holding" unless that IS the main action

Focus on:
- The task/action being performed (folding, opening, closing, pouring, cutting, assembling, etc.)
- The object being manipulated
- What changes from start to end of the action

Ignore:
- Other people in the frame
- Background details
- Static descriptions (e.g., "holding X" when the action is "folding X")

Format: Start with an action verb in present continuous tense (-ing form).

Good Examples (transformative actions):
- "Folding t-shirt on table" (transforms: unfolded → folded)
- "Opening and closing container lid" (transforms: closed → open → closed)
- "Pouring liquid into cup" (transforms: empty cup → filled cup)
- "Cutting vegetables on board" (transforms: whole → chopped)
- "Assembling furniture pieces" (transforms: parts → assembled)

Bad Examples (too static):
- "Holding t-shirt flat" ❌ → Should be "Folding t-shirt"
- "Holding container with lid" ❌ → Should be "Opening container lid"

Your description:"""
        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=[prompt, image])
        return response.text.strip()
    
    def annotate_frames(
        self, 
        frame_paths: List[Dict[str, Any]],
        pose_flows: Optional[List[Dict[str, Any]]] = None,
        object_interactions: Optional[List[Dict[str, Any]]] = None,
        pose_estimates: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Annotate frames with robust scene change detection, idle detection, and repetition handling.
        
        Args:
            frame_paths: List of frame metadata dicts
            pose_flows: Optional list of pose flow data for idle detection
            object_interactions: Optional list of object interaction data for idle detection
            pose_estimates: Optional list of pose estimates for hand visibility check
        
        Returns:
            List of segment annotations (idle segments filtered out if remove_idle_segments=True)
        """
        if not frame_paths:
            return []
        
        # Convert lists to dicts for faster lookup
        pose_flows_dict = {flow["frame_index"]: flow for flow in (pose_flows or [])} if pose_flows else {}
        interactions_dict = {inter["frame_index"]: inter for inter in (object_interactions or [])} if object_interactions else {}
        pose_estimates_dict = {pose["frame_index"]: pose for pose in (pose_estimates or [])} if pose_estimates else {}
        
        # First pass: extract all embeddings
        print("Extracting embeddings for all frames...")
        all_embeddings = []
        frame_metadata = []
        
        for frame_info in frame_paths:
            frame_index = frame_info.get("frame_index")
            if frame_index is None:
                continue
            
            frame_path = frame_info["frame_path"]
            embedding = self.embed_frame(frame_path)
            
            all_embeddings.append(embedding)
            frame_metadata.append({
                "frame_index": frame_index,
                "timestamp": frame_info.get("timestamp"),
                "frame_path": frame_path,
            })
            
            if frame_index % 30 == 0:
                print(f"  Processed frame {frame_index}")
        
        # Second pass: segment with robust boundary detection
        print("Performing robust scene segmentation...")
        segments = []
        current_segment_indices = [0]  # Start with first frame
        
        for i in range(1, len(all_embeddings)):
            if self.should_create_boundary(i, all_embeddings, current_segment_indices, frame_metadata):
                # Refine the boundary to find exact frame where change occurred
                refined_boundary = self.refine_boundary(
                    coarse_boundary_frame=i,
                    previous_segment_frame=current_segment_indices[-1],
                    frame_metadata=frame_metadata
                )
                
                # Add any frames between last segment frame and refined boundary to current segment
                for idx in range(current_segment_indices[-1] + 1, refined_boundary):
                    if idx < len(all_embeddings):
                        current_segment_indices.append(idx)
                
                # Finalize current segment
                segments.append(current_segment_indices)
                
                # Start new segment at refined boundary
                current_segment_indices = [refined_boundary]
            else:
                current_segment_indices.append(i)
        
        # Don't forget the last segment
        if current_segment_indices:
            segments.append(current_segment_indices)
        
        print(f"Created {len(segments)} coarse segments")
        
        # Third pass: detect idle segments and repetitions
        print("Detecting idle segments and repetitions...")
        refined_segments = []
        idle_count = 0
        repetition_count = 0
        
        for seg_idx, segment_indices in enumerate(segments):
            # Check if idle
            is_idle, avg_velocity, has_interactions, hands_visible, internal_similarity = self.detect_idle_segment(
                segment_indices, frame_metadata, pose_flows_dict, interactions_dict, pose_estimates_dict, all_embeddings
            )
            
            if is_idle:
                frame_start = frame_metadata[segment_indices[0]]["frame_index"]
                frame_end = frame_metadata[segment_indices[-1]]["frame_index"]
                if hands_visible and internal_similarity >= 0.75:
                    # Only log if not already logged in detect_idle_segment
                    print(f"  Segment {seg_idx} (frames {frame_start}-{frame_end}) is IDLE (velocity={avg_velocity:.4f})")
                # else: already logged in detect_idle_segment
                idle_count += 1
                
                if not self.remove_idle_segments:
                    # Keep idle segments but mark them
                    refined_segments.append({
                        "segment_indices": segment_indices,
                        "is_idle": True,
                        "metadata": {
                            "avg_velocity": avg_velocity,
                            "has_interactions": has_interactions,
                            "hands_visible": hands_visible,
                            "internal_similarity": internal_similarity
                        }
                    })
                # If remove_idle_segments=True, skip this segment
                continue
            
            # Check for repetitions
            instances = self.detect_repetitive_instances(
                segment_indices, all_embeddings, frame_metadata
            )
            
            if len(instances) > 1:
                frame_start = frame_metadata[segment_indices[0]]["frame_index"]
                frame_end = frame_metadata[segment_indices[-1]]["frame_index"]
                print(f"  Segment {seg_idx} (frames {frame_start}-{frame_end}) split into {len(instances)} instances")
                repetition_count += len(instances) - 1
            
            # Add all instances as separate segments
            for instance_indices in instances:
                refined_segments.append({
                    "segment_indices": instance_indices,
                    "is_idle": False,
                    "metadata": {}
                })
        
        print(f"Filtered out {idle_count} idle segments, detected {repetition_count} repetitions")
        print(f"Final segment count: {len(refined_segments)}")
        
        # Fourth pass: finalize segments with text annotations
        segment_annotations = []
        for refined_seg in refined_segments:
            segment_annotations.append(
                self._finalize_segment(
                    refined_seg["segment_indices"], 
                    frame_metadata, 
                    len(segment_annotations),
                    is_idle=refined_seg["is_idle"],
                    metadata=refined_seg["metadata"]
                )
            )
        
        return segment_annotations
    
    def _finalize_segment(
        self, 
        segment_indices: List[int], 
        frame_metadata: List[Dict[str, Any]], 
        segment_idx: int,
        is_idle: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Finalize a segment with text annotation.
        Returns segment metadata with text description.
        """
        frame_start_idx = segment_indices[0]
        frame_end_idx = segment_indices[-1]
        
        frame_start = frame_metadata[frame_start_idx]["frame_index"]
        frame_end = frame_metadata[frame_end_idx]["frame_index"]
        timestamp_start = frame_metadata[frame_start_idx]["timestamp"]
        timestamp_end = frame_metadata[frame_end_idx]["timestamp"]
        
        # Use middle frame for text annotation
        mid_idx = segment_indices[len(segment_indices) // 2]
        mid_frame_path = frame_metadata[mid_idx]["frame_path"]
        
        # Handle idle segments differently
        if is_idle:
            frame_annotation = "Idle - hands visible, no task activity"
        else:
            print(f"Generating description for segment {segment_idx} (frames {frame_start}-{frame_end})...")
            frame_annotation = self.generate_text_from_image(mid_frame_path)
        
        result = {
            "segment_index": segment_idx,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "text_description": frame_annotation,
            "key_frame_path": mid_frame_path,
            "is_idle": is_idle,
        }
        
        # Add metadata if provided
        if metadata:
            result["metadata"] = metadata
        
        return result

