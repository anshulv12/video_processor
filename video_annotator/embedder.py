from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from typing import List, Dict, Any
import google.generativeai as genai
import base64
from extracter import FrameExtractor

class FrameAnnotator:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", similarity_threshold: float = 0.7):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.similarity_threshold = similarity_threshold
        self.model.eval()

    def embed_frame(self, frame_path: str) -> np.ndarray:
        image = Image.open(frame_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs[0].cpu().numpy()
    
    def is_similar(self, image_embedding_a: np.ndarray, image_embedding_b: np.ndarray) -> float:
        return np.dot(image_embedding_a, image_embedding_b) / (np.linalg.norm(image_embedding_a) * np.linalg.norm(image_embedding_b)) >= self.similarity_threshold
    
    def generate_text_from_embedding(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{encoded_string}"
        prompt = f"Generate a text description of the image: {image_data}"
        model = genai.GenerativeModel(model="gemini-2.5-flash")
        response = model.generate_content(prompt=prompt)
        return response.text
    
    def annotate_frames(self, frame_paths: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
        cur_segment = [] # (timestamp, frame_path, image_embedding)
        segment_annotations = [] # (frame_start, frame_end, text_description)
        frame_paths.append({"timestamp": 0, "frame_path": None})
        for i, frame_info in frame_paths:
            timestamp, frame_path = frame_info['timestamp'], frame_info['frame_path']
            image_embedding = self.embed_frame(frame_path) if i < len(frame_paths) - 1 else None

            if cur_segment and i < len(frame_paths) - 1 and self.is_similar(cur_segment[-1][2], image_embedding):
                cur_segment.append((timestamp, frame_path, image_embedding))
            else:
                _, mid_frame_path, _ = cur_segment[len(cur_segment) // 2]
                timestamp_start = cur_segment[0][0]
                timestamp_end = cur_segment[-1][0]
                frame_annotation = self.generate_text_from_embedding(mid_frame_path)
                
                segment_annotations.append((timestamp_start, timestamp_end, frame_annotation))

                cur_segment = [(timestamp, frame_path, image_embedding)]

        return segment_annotations



        
if __name__ == "__main__":
    FrameExtractor = FrameExtractor("frames")
    frame_paths = FrameExtractor.extract_frames("video_data/0.mp4")
    annotator = FrameAnnotator()
    annotations = annotator.annotate_frames(frame_paths)
    print(annotations)
    