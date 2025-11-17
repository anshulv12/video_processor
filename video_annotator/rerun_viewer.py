#!/usr/bin/env python3
"""Use the MediaPipe Hands solution to detect and track hands in video."""

from __future__ import annotations

import argparse
import logging
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb

DESCRIPTION = """# Hand tracking with MediaPipe Hands

The 3D view shows:
- Colored hand landmarks (wrist, joints, fingertips)
- Hands in their natural POV orientation
- Skeleton connections between landmarks

Color scheme:
- 🟡 Yellow: Wrist
- 🔵 Cyan: Base knuckles
- 🟠 Orange: Middle joints
- 🟢 Green: Outer joints  
- 🟣 Magenta: Fingertips

Tips:
- Set background to black (0, 0, 0) in View settings for better contrast
- Both 2D (on video) and 3D visualizations update in real-time
"""


def get_landmark_colors():
    """
    Assign colors to each of the 21 hand landmarks based on their type.
    
    Landmark indices:
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, TIP)
    5-8: Index finger (MCP, PIP, DIP, TIP)
    9-12: Middle finger (MCP, PIP, DIP, TIP)
    13-16: Ring finger (MCP, PIP, DIP, TIP)
    17-20: Pinky (MCP, PIP, DIP, TIP)
    """
    colors = []
    
    # Color scheme
    WRIST_COLOR = (255, 255, 0)        # Yellow
    BASE_COLOR = (0, 255, 255)         # Cyan (base knuckles/MCP)
    MIDDLE_COLOR = (255, 128, 0)       # Orange (middle joints)
    OUTER_COLOR = (128, 255, 0)        # Light green (outer joints)
    TIP_COLOR = (255, 0, 255)          # Magenta (fingertips)
    
    for i in range(21):
        if i == 0:
            # Wrist
            colors.append(WRIST_COLOR)
        elif i in [1, 5, 9, 13, 17]:
            # Base knuckles (CMC for thumb, MCP for fingers)
            colors.append(BASE_COLOR)
        elif i in [2, 6, 10, 14, 18]:
            # Middle joints (MCP for thumb, PIP for fingers)
            colors.append(MIDDLE_COLOR)
        elif i in [3, 7, 11, 15, 19]:
            # Outer joints (IP for thumb, DIP for fingers)
            colors.append(OUTER_COLOR)
        elif i in [4, 8, 12, 16, 20]:
            # Fingertips
            colors.append(TIP_COLOR)
    
    return colors


def track_hands(video_path: str, *, max_frame_count: int | None, model_complexity: int = 1, output_rrd: str | None = None) -> None:
    """Track hands in a video using MediaPipe Hands.
    
    Args:
        video_path: Path to video file
        max_frame_count: Optional limit on frames to process
        model_complexity: MediaPipe model complexity (0 or 1)
        output_rrd: Optional path to save .rrd recording file
    """
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), static=True)

    # Set up annotation context for hands with colored connections
    hand_landmark_connections = mp_hands.HAND_CONNECTIONS
    
    # Create annotation contexts for left and right hands
    rr.log(
        "/",
        rr.AnnotationContext([
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=1, color=(255, 255, 255)),  # Default color
                keypoint_connections=hand_landmark_connections,
            ),
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=2, color=(255, 255, 255)),  # Default color
                keypoint_connections=hand_landmark_connections,
            ),
        ]),
        static=True,
    )
    
    rr.log("hands", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    # Define landmark colors by type
    # MediaPipe Hands has 21 landmarks: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    landmark_colors = get_landmark_colors()

    with closing(VideoSource(video_path)) as video_source:
        for idx, bgr_frame in enumerate(video_source.stream_bgr()):
            if max_frame_count is not None and idx >= max_frame_count:
                break

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(bgr_frame.data, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            rr.set_time("time", duration=bgr_frame.time)
            rr.set_time("frame_idx", sequence=bgr_frame.idx)

            results = hands.process(rgb_frame)
            h, w, _ = bgr_frame.data.shape

            # Log the original video frame in 2D views
            rr.log("video/rgb", rr.Image(rgb_frame).compress(jpeg_quality=75))

            # Process hand landmarks if detected
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # Get hand label (Left or Right)
                    hand_label = handedness.classification[0].label
                    class_id = 1 if hand_label == "Left" else 2
                    
                    # Log 2D points on the video
                    landmark_positions_2d = np.array([
                        (w * lm.x, h * lm.y) for lm in hand_landmarks.landmark
                    ])
                    
                    rr.log(
                        f"video/hands/{hand_label}",
                        rr.Points2D(
                            landmark_positions_2d,
                            class_ids=class_id,
                            keypoint_ids=list(range(len(hand_landmarks.landmark))),
                            colors=landmark_colors,
                            radii=3,
                        ),
                    )
                    
                    # Log 3D points in world coordinates (flip X for performer POV)
                    landmark_positions_3d = np.array([
                        (-lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark  # Negative X for mirror effect
                    ])
                    
                    rr.log(
                        f"hands/{hand_label}",
                        rr.Points3D(
                            landmark_positions_3d,
                            class_ids=class_id,
                            keypoint_ids=list(range(len(hand_landmarks.landmark))),
                            colors=landmark_colors,
                            radii=0.01,
                        ),
                    )

    hands.close()


@dataclass
class VideoFrame:
    data: cv2.typing.MatLike
    time: float
    idx: int


class VideoSource:
    def __init__(self, path: str) -> None:
        self.capture = cv2.VideoCapture(path)

        if not self.capture.isOpened():
            logging.error("Couldn't open video at %s", path)

    def close(self) -> None:
        self.capture.release()

    def stream_bgr(self) -> Iterator[VideoFrame]:
        while self.capture.isOpened():
            idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            is_open, bgr = self.capture.read()
            time_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)

            if not is_open:
                break

            yield VideoFrame(data=bgr, time=time_ms * 1e-3, idx=idx)


def process_video_with_rerun(video_path: str, output_rrd: str, max_frame: int | None = None, model_complexity: int = 1) -> None:
    """Process a video and save rerun recording to file.
    
    Args:
        video_path: Path to input video
        output_rrd: Path to save .rrd recording file
        max_frame: Optional frame limit
        model_complexity: MediaPipe model complexity
    """
    # Initialize rerun to save to file
    rr.init("hand_tracking", recording_id=f"video_{Path(video_path).stem}")
    rr.save(output_rrd)
    
    # Set up minimal dark blueprint with just 2D and 3D views
    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin="video", 
                name="Video with Hand Landmarks",
            ),
            rrb.Spatial3DView(
                origin="hands", 
                name="3D Hand Landmarks",
            ),
            column_shares=[1, 1],
        )
    )
    
    logging.info(f"Processing video: {video_path}")
    track_hands(video_path, max_frame_count=max_frame, model_complexity=model_complexity, output_rrd=output_rrd)
    logging.info(f"Saved recording to: {output_rrd}")


def main() -> None:
    # Ensure the logging gets written to stderr:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel("INFO")

    parser = argparse.ArgumentParser(description="Uses the MediaPipe Hands solution to track hands in video.")
    parser.add_argument(
        "--video-path",
        type=str,
        default="video_data/0.mp4",
        help="Path to the video file to process.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1],
        help="Complexity of the hand landmark model: 0 or 1. Higher is more accurate but slower.",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        help="Stop after processing this many frames. If not specified, will run until interrupted.",
    )
    parser.add_argument(
        "--output-rrd",
        type=str,
        help="Path to save .rrd recording file. If not specified, opens viewer instead.",
    )
    rr.script_add_args(parser)

    args = parser.parse_args()
    
    # Use the provided video path directly
    video_path = args.video_path
    if not Path(video_path).exists():
        logging.error(f"Video file not found: {video_path}")
        return

    # If output path specified, save to file
    if args.output_rrd:
        process_video_with_rerun(video_path, args.output_rrd, args.max_frame, args.model_complexity)
    else:
        # Otherwise, open viewer
        rr.script_setup(
            args,
            "rerun_hand_tracking",
            default_blueprint=rrb.Horizontal(
                rrb.Spatial2DView(
                    origin="video", 
                    name="Video with Hand Landmarks",
                ),
                rrb.Spatial3DView(
                    origin="hands", 
                    name="3D Hand Landmarks",
                ),
                column_shares=[1, 1],
            ),
        )
        
        logging.info(f"Processing video: {video_path}")
        track_hands(video_path, max_frame_count=args.max_frame, model_complexity=args.model_complexity)
        
        rr.script_teardown(args)


if __name__ == "__main__":
    main()
