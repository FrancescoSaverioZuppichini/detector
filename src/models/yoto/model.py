from torch import nn

from src.types import Backbone, Neck

from .head import Head


class YOTO(nn.Module):
    def __init__(self, backbone: Backbone, neck: Neck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        features = self.neck(features)

        return features


class YOTOForObjectDetection(nn.Module):
    def __init__(self, backbone: Backbone, neck: Neck, head: Head):
        super().__init__()
        self.model = YOTO(backbone, neck)
        self.head = head

    def forward(self, pixel_values):
        features = self.model(pixel_values)
        outputs = self.head(features)

        return outputs
