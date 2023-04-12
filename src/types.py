from torch import nn
from torch import Tensor
from typing import List

class Backbone(nn.Module):
    def forward(self, pixels: Tensor) -> List[Tensor]:
        raise NotImplemented
    
class Neck(nn.Module):
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        raise NotImplemented