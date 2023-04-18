from typing import Tuple

import torch
from torch import nn

from src.types import Features

from ..types import Neck
from .common import Conv2dLayerNormGELULayer, ConvTranspose2dLayerNormGELULayer


class SimpleFPN(Neck):
    def __init__(
        self,
        in_channels: int = 768,
        out_channels: Tuple[int] = (256, 256, 256, 256),
        strides: Tuple[int] = (4, 8, 16, 32),
        target_stride: int = 16,
    ):
        super().__init__()
        self.down_4 = nn.Sequential(
            ConvTranspose2dLayerNormGELULayer(
                in_channels, out_channels[0], kernel_size=2, stride=4
            ),
            Conv2dLayerNormGELULayer(out_channels[0], out_channels[0], 1),
        )
        self.down_8 = nn.Sequential(
            ConvTranspose2dLayerNormGELULayer(
                in_channels, out_channels[1], kernel_size=2, stride=2
            ),
            Conv2dLayerNormGELULayer(out_channels[1], out_channels[1], 1),
        )
        self.down_16 = nn.Sequential(
            Conv2dLayerNormGELULayer(in_channels, out_channels[2])
        )
        self.down_32 = nn.Sequential(
            Conv2dLayerNormGELULayer(
                in_channels, out_channels[3], kernel_size=2, stride=2
            ),
            Conv2dLayerNormGELULayer(out_channels[3], out_channels[3], kernel_size=1),
        )

    def forward(self, features: Features) -> Features:
        # simple neck only uses the last one
        feature = features[-1]
        x_down_4 = self.down_4(feature)
        x_down_8 = self.down_8(feature)
        x_down_16 = self.down_16(feature)
        x_down_32 = self.down_32(feature)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


if __name__ == "__main__":
    from torchinfo import summary

    # using always the first feature
    fpn = SimpleFPN()
    summary(fpn, input_size=(1, 768, 40, 40))
    outs = fpn([torch.randn((1, 768, 40, 40))])
    print(fpn)
    print([f.shape for f in outs])
