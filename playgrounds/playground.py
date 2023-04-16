from src.tasks.head import Head
from src.tasks.loss import OneNetLoss
import torch
from src.tasks.object_detection.matcher import MinCostMatcher
from src.nn.fpn import SimpleFPN
from src.nn.vit import ViT
from einops import rearrange
from src.nn.adapters import ViTAdapterForNeck
from src.nn.yoto import YOTOForObjectDetection

backbone = ViTAdapterForNeck(ViT(224, patch_size=16, width=768, layers=4, heads=8, output_dim=512))
fpn = SimpleFPN(in_channels=768)
head = Head(256, channels=256, num_classes=2)
criterion = OneNetLoss(num_classes=2, matcher=MinCostMatcher())


yoto = YOTOForObjectDetection(backbone, fpn, head)
pixel_values = torch.randn((2, 3, 224, 224))
outs = yoto(pixel_values)

class_labels = torch.tensor(
    [
        [1, 0],
        [0, 0],
    ]
)
boxes_labels = torch.tensor(
    [
        [
            [0.1, 0.1, 0.3, 0.3],
            [0.2, 0.2, 0.4, 0.4],
        ],
        [[0.1, 0.1, 0.3, 0.3], [0, 0, 0, 0]],  # pad
    ]
)
mask_labels = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
print(class_labels.shape, boxes_labels.shape)

features = backbone(pixel_values)
print(features[0].shape, print(len(features)))
pyramids = fpn(features)
outs = head(pyramids)
losses = criterion(*outs, class_labels, boxes_labels, torch.tensor([640]), mask_labels)

print(losses)
