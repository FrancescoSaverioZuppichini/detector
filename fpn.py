from typing import Optional, List, Callable, Any, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from common import Conv2DNormGELULayer, ConvTranspose2dNormGELULayer


class SimpleFPN(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        out_channels: Tuple[int] = (256, 256, 256, 256),
        strides: Tuple[int] = (4, 8, 16, 32),
        target_stride: int = 16,
    ):
        super().__init__()
        self.down_4 = nn.Sequential(
            ConvTranspose2dNormGELULayer(
                in_channels, out_channels[0], kernel_size=2, stride=4
            ),
            Conv2DNormGELULayer(out_channels[0], out_channels[0], 1),
        )
        self.down_8 = nn.Sequential(
            ConvTranspose2dNormGELULayer(
                in_channels, out_channels[1], kernel_size=2, stride=2
            ),
            Conv2DNormGELULayer(out_channels[1], out_channels[1], 1),
        )
        self.down_16 = nn.Sequential(Conv2DNormGELULayer(in_channels, out_channels[2]))
        self.down_32 = nn.Sequential(
            Conv2DNormGELULayer(in_channels, out_channels[3], kernel_size=2, stride=2),
            Conv2DNormGELULayer(out_channels[3], out_channels[3], kernel_size=1),
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x: Tensor) -> List[Tensor]:
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


if __name__ == "__main__":
    from torchinfo import summary

    fpn = SimpleFPN()
    summary(fpn, input_size=(1, 768, 40, 40))
    outs = fpn(torch.randn((1, 768, 40, 40)))
    print(fpn)
    print([f.shape for f in outs])
