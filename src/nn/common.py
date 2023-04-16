from functools import partial
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class QuickGELU(nn.Module):
    """
    Copied from CLIP source code: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#L166
    """

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class ConvLayerNormGELULayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        conv_layer: Callable[[Any], nn.Module] = nn.Conv2d,
        *args,
        **kwargs
    ):
        super().__init__()
        self.conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
            *args,
            **kwargs
        )
        # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = QuickGELU()


Conv2dLayerNormGELULayer = ConvLayerNormGELULayer
ConvTranspose2dLayerNormGELULayer = partial(
    ConvLayerNormGELULayer, conv_layer=nn.ConvTranspose2d
)

class StackedConv2dLayers(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        conv_layer: Callable[[Any], nn.Module] = ConvLayerNormGELULayer,
        *additional_layers
    ):
        super().__init__()
        self.layers = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=3),
            *[
                conv_layer(out_channels, out_channels, kernel_size=3)
                for _ in range(depth - 1)
            ],
            *additional_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
