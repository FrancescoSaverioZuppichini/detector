import pytorch_lightning as pl
from src.loss import OneNetLoss
from src.data import ObjectDetectionData
from src.matcher import MinCostMatcher
import torch

class ObjectDetectionTask(pl.LightningModule):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.loss = OneNetLoss(num_classes, matcher=MinCostMatcher())

    def training_step(self, batch: ObjectDetectionData, batch_idx: int):
        outs = self.model(batch.pixel_values)
        print(batch.labels.dtype)
        loss = self.loss(*outs, batch.labels, batch.bboxes, batch.images_sizes)
        tot_loss = torch.tensor(.0)
        for loss in loss.values():
            tot_loss += loss
        print(tot_loss)
        return tot_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer