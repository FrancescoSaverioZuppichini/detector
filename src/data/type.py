import torch
from tensordict.prototype import tensorclass


@tensorclass
class ObjectDetectionData:
    image: torch.Tensor
    # format x1,y1,x2,y2
    bboxes: torch.Tensor
    labels: torch.Tensor
    images_sizes: torch.Tensor
