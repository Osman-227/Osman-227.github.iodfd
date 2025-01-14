import os
from typing import List, Optional, Union

import numpy as np
import torch

from norfair import Detection


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Check if the model exists, skip download if already present
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path '{model_path}' does not exist.")

        # Load custom YOLO model (replace the default loading code)
        try:
            self.model = torch.hub.load("ultralytics/yolo11s", "custom", path=model_path)
            self.model.to(device)
        except Exception as e:
            raise Exception(f"Failed to load model from {model_path}. Error: {e}")

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.4,
        image_size: int = 1080,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [
                    [detection_as_xywh[0].item(), detection_as_xywh[1].item()],
                    [detection_as_xywh[0].item(), detection_as_xywh[1].item()],
                ]
            )
            scores = np.array(
                [detection_as_xywh[4].item(), detection_as_xywh[4].item()]
            )
            norfair_detections.append(Detection(points=centroid, scores=scores))
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(Detection(points=bbox, scores=scores))

    return norfair_detections
