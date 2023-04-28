from functools import reduce

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchvision.ops.boxes import box_convert

from src.data.data import ObjectDetectionData

from .loss import OneNetLoss
from .matcher import MinCostMatcher
from .model import YOTOForObjectDetection


class YOTOForObjectDetectionTask(pl.LightningModule):
    def __init__(
        self,
        model: YOTOForObjectDetection,
        loss: OneNetLoss,
        learning_rate: float = 1e-3,
        transform: nn.Module = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.transform = transform
        self.learning_rate = learning_rate

    def training_step(self, batch: ObjectDetectionData, batch_idx: int):
        batch.images = batch.images.float().div_(255)
        batch.bboxes = box_convert(batch.bboxes, in_fmt="xywh", out_fmt="xyxy")
        if self.transform is not None:
            batch = self.transform(batch)
        class_logits, boxes_preds = self.model(batch.images)
        class_logits, boxes_preds = class_logits.float(), boxes_preds.float()
        loss = self.loss(
            class_logits,
            boxes_preds,
            batch.classes,
            batch.bboxes,
            batch.images_sizes,
            batch.mask_labels,
        )
        # reduce loss by summing
        tot_loss = torch.tensor(0.0, device=batch.images.device)
        for loss in loss.values():
            tot_loss += loss
        return tot_loss

    def forward(self, pixel_values):
        outs = self.model(pixel_values)
        return outs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer
