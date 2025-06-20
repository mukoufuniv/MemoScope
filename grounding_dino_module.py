import torch
import numpy as np
import cv2

try:
    from groundingdino.util.inference import load_model, load_image, predict
except ImportError as e:
    load_model = None
    load_image = None
    predict = None

class GroundingDINOModel:
    """Wrapper for Grounding DINO inference."""

    def __init__(self, config_path: str, weights_path: str, device: str | None = None):
        if load_model is None:
            raise ImportError(
                "GroundingDINO is not installed. Please install the groundingdino package to use this module."
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(config_path, weights_path)
        self.model.to(self.device)

    def predict_boxes(self, image: np.ndarray, prompt: str, box_threshold: float = 0.35,
                       text_threshold: float = 0.25):
        """Run prediction on a single image with a text prompt."""
        if load_image is None or predict is None:
            raise ImportError(
                "GroundingDINO is not installed. Please install the groundingdino package to use this module."
            )
        image_source, image = load_image(image)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        # convert box format xyxy-normalized -> absolute pixel coordinates
        h, w = image_source.shape[:2]
        boxes[:, 0::2] *= w
        boxes[:, 1::2] *= h
        boxes = boxes.astype(int)
        return boxes, logits, phrases

    @staticmethod
    def draw_boxes(image: np.ndarray, boxes: np.ndarray, phrases) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        for box, phrase in zip(boxes, phrases):
            x0, y0, x1, y1 = box
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                image,
                phrase,
                (x0, max(y0 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return image

