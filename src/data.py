from tensordict.prototype import tensorclass
import torch

@tensorclass
class ObjectDetectionData:
   pixel_values: torch.Tensor
   bboxes: torch.Tensor
   labels: torch.Tensor
   images_sizes: torch.Tensor