from src.head import Head
from src.loss import OneNetLoss
import torch
from src.matcher import MinCostMatcher
from src.fpn import SimpleFPN
from src.vit import CLIPViT
from einops import rearrange

backbone = CLIPViT(224, patch_size=16, width=768, layers=4, heads=8, output_dim=512)
fpn = SimpleFPN(in_channels=768)
head = Head(256, channels=256, num_classes=2)
criterion = OneNetLoss(num_classes=2, matcher=MinCostMatcher())


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

features = backbone(torch.randn((2, 3, 224, 224)))
print(features[0].shape)
features = [rearrange(f, "b (h w) d -> b d h w", h=224//16, w=224//16) for f in features]
pyramids = fpn(features)
outs = head(pyramids)
losses = criterion(*outs, class_labels, boxes_labels, mask_labels)

print(losses)
