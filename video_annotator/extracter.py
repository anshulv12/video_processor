import cv2
import os

class FrameExtractor:
    def __init__(self, output_dir: str, fps: int = 1):
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

    def extract_frames(self, video_path: str):
        video_cap = cv2.VideoCapture(video_path)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS) or 1.0
        frame_interval = int(max(1, video_fps / self.fps))
        
        frame_idx = 0
        frames_info = []
        while video_cap.isOpened():
            success, frame = video_cap.read()
            if not success:
                break
            if frame_idx % frame_interval == 0:
                frame_path = os.path.join(self.output_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame)
                frames_info.append({
                    "timestamp": frame_idx / video_fps,
                    "frame_path": frame_path,
                    "frame_index": frame_idx
                })
            frame_idx += 1
        video_cap.release()
        return frames_info
