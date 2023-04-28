import pytorch_lightning as pl
import sys
from src.models.yoto.task import YOTOForObjectDetectionTask
from src.data.data import ObjectDetectionData
from src.nn.fpn import SimpleFPN
from src.nn.vit import ViT
from einops import rearrange
from src.nn.adapters import ViTAdapterForNeck
from src.models.yoto.model import YOTOForObjectDetection
from src.models.yoto.head import Head
from src.models.yoto.loss import OneNetLoss
from src.models.yoto.matcher import MinCostMatcher
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from src.data.datasets.yolo import YOLODataset
from src.data.transform import Resize
from pathlib import Path
from typing import List
from tqdm import tqdm
from torchvision.ops.boxes import box_convert


num_classes = 5 + 1
size = (640, 480)
vit = ViT(size, patch_size=16, width=768, layers=12, heads=12, output_dim=512)
for param in vit.parameters():
    param.requires_grad = False

backbone = ViTAdapterForNeck(vit)
fpn = SimpleFPN(in_channels=768)
head = Head(256, channels=256, num_classes=num_classes)


criterion = OneNetLoss(num_classes, matcher=MinCostMatcher())
yoto = YOTOForObjectDetection(backbone, fpn, head)

# yoto = torch.compile(yoto)
transform = Resize(size)
# transform = torch.compile(transform)
# yoto.model.backbone.vit.positional_embedding.data = yoto.model.backbone.vit.interpolate_pos_encoding(640,640)
task = YOTOForObjectDetectionTask(yoto, criterion, transform=transform)


class Collate(nn.Module):
    def __init__(self, transform=None, device=None):
        super().__init__()
        self.transform = transform
        # self.device = torch.device(device)

    def __call__(self, x: List[ObjectDetectionData]):
        images = []
        bboxes = []
        classes = []
        images_sizes = []
        mask_labels = []

        for data in x:
            images.append(data.images)
            bboxes.append(data.bboxes)
            classes.append(data.classes)
            images_sizes.append(data.images_sizes)
            mask_labels.append(data.mask_labels)

        data = ObjectDetectionData(
            torch.cat(images, dim=0),
            torch.cat(bboxes, dim=0),
            torch.cat(classes, dim=0),
            torch.cat(images_sizes, dim=0),
            torch.cat(mask_labels, dim=0),
            batch_size=[len(x)],
            device=x[0].device,
        )
        data = data.contiguous()
        # if self.device.type == "cuda":
        #     out = data.pin_memory()
        # if self.device:
        #     # move data to gpu
        #     data = data.to(self.device, non_blocking=True)

        # data.images = data.images.float().div_(255)
        # if self.transform:
        #     # apply transforms on gpu
        #     data = self.transform(data)
        return data


ds = YOLODataset(
    Path("/home/zuppif/Documents/neatly/detector/benchmarks/data/datasets/lettuce-pallets-640/train"), padding=True
)
train_dl = DataLoader(
    ds, batch_size=32, collate_fn=Collate(), num_workers=8, pin_memory=True, persistent_workers=True
)

# torch.set_float32_matmul_precision('high')
# task = torch.compile(task)
trainer = pl.Trainer(accelerator="gpu", precision='16')
trainer.fit(model=task, train_dataloaders=train_dl)
