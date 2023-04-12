from .types import Backbone, Neck
from torch import nn


class YOTO:
    def __init__(self, backbone: Backbone, neck: Neck):
        self.backbone = backbone
        self.neck = neck

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        features = self.neck(features)

        return features


class YOTOForObjectDetection(YOTO):
    def __init__(self, backbone: Backbone, neck: Neck, head: nn.Module):
        super().__init__(backbone, neck)
        self.head = head

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        refined_features = self.neck(features)
        outputs = self.head(refined_features)

        return outputs
