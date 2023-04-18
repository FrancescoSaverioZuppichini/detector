from functools import reduce

import pytorch_lightning as pl
import torch
from torch import optim

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
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate

    def training_step(self, batch: ObjectDetectionData, batch_idx: int):
        outs = self.model(batch.pixel_values)
        loss = self.loss(*outs, batch.labels, batch.bboxes, batch.images_sizes)
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
