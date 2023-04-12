from torch import nn
from einops import rearrange
from src.vit import ViT


class ViTAdapterForNeck(nn.Module):
    def __init__(self, vit: ViT) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, *args, **kwargs):
        print(self.vit.input_resolution)
        features = self.vit(*args, **kwargs)
        features = [
            rearrange(
                feature,
                "b (h w) c -> b c h w",
                h=self.vit.input_resolution // self.vit.patch_size,
                w=self.vit.input_resolution // self.vit.patch_size,
            ).contiguous()
            for feature in features
        ]
        return features
