"""Model to detect objects we specified in an image."""

from semantic_inference.config import Config, config_field
from semantic_inference.models.patch_extractor import get_image_preprocessor
from semantic_inference.models.patch_extractor import center_crop

import torch
import torch.nn.functional as F
from torch import nn

from typing import Any
import dataclasses
from dataclasses import dataclass
import numpy as np


def _map_opt(values, f):
    return {k: v if v is None or isinstance(v, list) == True else f(v) for k, v in values.items()}


@dataclass
class DetectionResults:
    """Openset Detection Results."""
    masks: torch.Tensor
    boxes: torch.Tensor
    confs: torch.Tensor
    labels: torch.Tensor
    mapping: list[str]
    image_embedding: torch.Tensor
    
    def cpu(self):
        """Move results to CPU."""
        values = dataclasses.asdict(self)
        return DetectionResults(**_map_opt(values, lambda v: v.cpu()))

    def to(self, *args, **kwargs):
        """Forward to to all tensors."""
        values = dataclasses.asdict(self)
        return DetectionResults(**_map_opt(values, lambda v: v.to(*args, **kwargs)))

    def __len__(self):
        """Get number of detected objects."""
        return len(self.boxes)


@dataclass
class OpensetDetectorConfig(Config):
    """Main config for openset detector."""
    clip_model: Any = config_field("clip", default="clip")
    detection: Any = config_field("detection", default="yoloe")

class OpensetDetector(nn.Module):
    # Detect objects in an image and encode the image with CLIP (for room detection)
    def __init__(self, config):
        """Construct an openset detector."""
        super(OpensetDetector, self).__init__()
        # for detecting model device
        self._canary_param = nn.Parameter(torch.empty(0))

        self.config = config
        self.detector = self.config.detection.create()
        self.encoder = self.config.clip_model.create()
        self.preprocess = get_image_preprocessor(self.encoder.input_size)
    
    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpensetDetectorConfig()
        config.update(kwargs)
        return cls(config)
    
    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device
    
    @torch.no_grad()
    def forward(self, rgb_img, is_rgb_order=True):
        """
        Detect objects in an image.

        Args:
            rgb_img (np.ndarray): uint8 image of shape (R, C, 3) in rgb order

        Returns:
            Results: Detected objects with masks and features
        """
        img_np = rgb_img if is_rgb_order else rgb_img[:, :, ::-1].copy()
        boxes, masks, mapping = self.detector(img_np, device=self.device)
        if boxes is None or masks is None:
            return None
        
        img = torch.from_numpy(img_np).to(self.device)
        img = img.permute((2, 0, 1))
        img = self.preprocess(img)
        clip_img = center_crop(img, self.encoder.input_size)
        img_embedding = torch.squeeze(self.encoder(clip_img.unsqueeze(0)))

        return DetectionResults(masks, boxes.xyxy, boxes.conf, boxes.cls, mapping, img_embedding)
