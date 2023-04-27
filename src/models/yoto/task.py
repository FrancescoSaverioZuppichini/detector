from functools import reduce

import pytorch_lightning as pl
import torch
from torch import optim

from src.data.data import ObjectDetectionData
from torch import nn
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
        batch = batch.cuda()
        if self.transform is not None:
            batch = self.transform(batch)
        outs = self.model(batch.images)
        loss = self.loss(*outs, batch.classes, batch.bboxes, batch.images_sizes, batch.mask_labels)
        # reduce loss by summing
        tot_loss = torch.tensor(0.0)
        for loss in loss.values():
            tot_loss += loss
        return tot_loss

    def forward(self, pixel_values):
        outs = self.model(pixel_values)
        return outs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer
