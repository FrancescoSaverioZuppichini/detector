from typing import List

from torch import Tensor, nn

Features = List[Tensor]

class Backbone(nn.Module):
    def forward(self, pixels: Tensor) -> List[Tensor]:
        raise NotImplemented


class Neck(nn.Module):
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        raise NotImplemented
