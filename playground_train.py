import pytorch_lightning as pl
from src.tasks.object_detection.task import ObjectDetectionTask
from src.data import ObjectDetectionData
from src.fpn import SimpleFPN
from src.vit import ViT
from einops import rearrange
from src.nn.adapters import ViTAdapterForNeck
from src.yoto import YOTOForObjectDetection
from src.head import Head
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

num_classes = 2
vit = ViT(224, patch_size=16, width=768, layers=4, heads=8, output_dim=512)
for param in vit.parameters():
    param.requires_grad = False
    
backbone = ViTAdapterForNeck(vit)
fpn = SimpleFPN(in_channels=768)
head = Head(256, channels=256, num_classes=2)


yoto = YOTOForObjectDetection(backbone, fpn, head)
task = ObjectDetectionTask(yoto, num_classes=2)

images = torch.randn(2, 3, 224, 224)
bboxes = torch.randn(2, 1, 4)
labels = torch.randint(2, (2,1), dtype=torch.int64)
images_sizes = torch.tensor([224, 224])

summary(yoto, input_data=images)


# data = ObjectDetectionData(images, bboxes, labels, images_sizes, batch_size=[2], device="cpu")


# class Collate(nn.Module):
#     def __init__(self, transform=None, device=None):
#         super().__init__()
#         self.transform = transform
#         self.device = torch.device(device)

#     def __call__(self, x: ObjectDetectionData):
#         out = x
#         # move data to RAM
#         # if self.device.type == "cuda":
#         #     out = x.apply(lambda x: x.as_tensor()).pin_memory()
#         # else:
#         #     out = x.apply(lambda x: x.as_tensor())
#         # if self.device:
#         #     # move data to gpu
#         #     out = out.to(self.device)
#         # if self.transform:
#         #     # apply transforms on gpu
#         #     out.images = self.transform(out.images)
#         return out

# train_dl = DataLoader(
#     data,
#     batch_size=2,
#     collate_fn=Collate(None, "cpu"),
# )
# print(data)
# trainer = pl.Trainer(accelerator="cpu")
# trainer.fit(model=task, train_dataloaders=train_dl)