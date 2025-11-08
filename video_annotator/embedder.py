from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from google import genai
from extracter import FrameExtractor
import os
from dotenv import load_dotenv

load_dotenv()
class FrameAnnotator:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", similarity_threshold: float = 0.7):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.similarity_threshold = similarity_threshold
        self.model.eval()
        self.client = genai.Client()

    def embed_frame(self, frame_path: str) -> np.ndarray:
        image = Image.open(frame_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs[0].cpu().numpy()
    
    def is_similar(self, image_embedding_a: np.ndarray, image_embedding_b: np.ndarray) -> bool:
        similarity = np.dot(image_embedding_a, image_embedding_b) / (np.linalg.norm(image_embedding_a) * np.linalg.norm(image_embedding_b))
        return similarity >= self.similarity_threshold
    
    def generate_text_from_image(self, image_path: str) -> str:
        image = Image.open(image_path)
        prompt = "Generate a concise text description of what is happening in this image."
        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=[prompt, image])
        return response.text
    
    def annotate_frames(self, frame_paths: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
        cur_segment = [] # (timestamp, frame_path, image_embedding)
        segment_annotations = [] # (frame_start, frame_end, text_description)
        
        for i, frame_info in enumerate(frame_paths):
            timestamp, frame_path = frame_info['timestamp'], frame_info['frame_path']
            image_embedding = self.embed_frame(frame_path)

            if cur_segment and self.is_similar(cur_segment[-1][2], image_embedding):
                cur_segment.append((timestamp, frame_path, image_embedding))
            else:
                if cur_segment:
                    _, mid_frame_path, _ = cur_segment[len(cur_segment) // 2]
                    timestamp_start = cur_segment[0][0]
                    timestamp_end = cur_segment[-1][0]
                    frame_annotation = self.generate_text_from_image(mid_frame_path)
                    segment_annotations.append((timestamp_start, timestamp_end, frame_annotation))
                
                cur_segment = [(timestamp, frame_path, image_embedding)]
        
        if cur_segment:
            _, mid_frame_path, _ = cur_segment[len(cur_segment) // 2]
            timestamp_start = cur_segment[0][0]
            timestamp_end = cur_segment[-1][0]
            frame_annotation = self.generate_text_from_image(mid_frame_path)
            segment_annotations.append((timestamp_start, timestamp_end, frame_annotation))

        return segment_annotations

        
if __name__ == "__main__":
    extractor = FrameExtractor("frames")
    frame_paths = extractor.extract_frames("video_data/0.mp4")
    annotator = FrameAnnotator()
    annotations = annotator.annotate_frames(frame_paths)
    print(annotations)
    