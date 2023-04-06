from typing import Optional, List, Callable, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from functools import partial


class ConvNormGELULayer(nn.Sequential):
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
        self.act = nn.GELU()


Conv2DNormGELULayer = ConvNormGELULayer
ConvTranspose2dNormGELULayer = partial(ConvNormGELULayer, conv_layer=nn.ConvTranspose2d)
