from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
class FrameAnnotator:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", similarity_threshold: float = 0.9):
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
        print(similarity)
        return similarity >= self.similarity_threshold
    
    def generate_text_from_image(self, image_path: str) -> str:
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
    
    def annotate_frames(self, frame_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cur_segment: List[Dict[str, Any]] = []
        segment_annotations: List[Dict[str, Any]] = []
        
        for frame_info in frame_paths:
            frame_index = frame_info.get("frame_index")
            if frame_index is None:
                continue
            timestamp = frame_info.get("timestamp")
            frame_path = frame_info["frame_path"]
            image_embedding = self.embed_frame(frame_path)
            print("frame_index", frame_index)
            if cur_segment and self.is_similar(cur_segment[-1]["embedding"], image_embedding):
                cur_segment.append({
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "frame_path": frame_path,
                    "embedding": image_embedding,
                })
            else:
                if cur_segment:
                    segment_annotations.append(self._finalize_segment(cur_segment, len(segment_annotations)))
                
                cur_segment = [{
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "frame_path": frame_path,
                    "embedding": image_embedding,
                }]
        
        if cur_segment:
            segment_annotations.append(self._finalize_segment(cur_segment, len(segment_annotations)))

        return segment_annotations

    def _finalize_segment(self, segment: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
        frame_start = segment[0]["frame_index"]
        frame_end = segment[-1]["frame_index"]
        timestamp_start = segment[0].get("timestamp")
        timestamp_end = segment[-1].get("timestamp")
        mid_frame_path = segment[len(segment) // 2]["frame_path"]
        frame_annotation = self.generate_text_from_image(mid_frame_path)

        return {
            "segment_index": index,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "text_description": frame_annotation,
            "key_frame_path": mid_frame_path,
        }
