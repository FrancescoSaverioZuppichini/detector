from einops import rearrange
from torch import nn
import torch

from src.nn.vit import ViT
from src.types import Features


class ViTAdapterForNeck(nn.Module):
    def __init__(self, vit: ViT) -> None:
        super().__init__()
        self.vit = vit

    
    def forward(self, *args, **kwargs) -> Features:
        with torch.no_grad():
            self.vit = self.vit.eval()
            features = self.vit(*args, **kwargs)
            features = [
                rearrange(
                    feature,
                    "b (h w) c -> b c h w",
                    h=self.vit.input_resolution[0] // self.vit.patch_size,
                    w=self.vit.input_resolution[1] // self.vit.patch_size,
                ).contiguous()
                for feature in features
            ]
        return features
