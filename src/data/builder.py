import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from tensordict import MemmapTensor, TensorDict
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from src.logger import logger

from .data import ObjectDetectionData
from .datasets.yolo import YOLODataset


class ObjectDetectionDatasetBuilder:
    def __init__(self, dst: Path):
        self.dst = dst

    def _build(
        self, dataset: YOLODataset, batch_size: int, num_workers: Optional[int] = None
    ) -> ObjectDetectionData:
        self.dst.mkdir(exist_ok=True, parents=True)
        logger.info(f"🛠️ Building dataset using {dataset.__class__.__name__}")
        dataset_size = len(dataset)
        batch_size = 8 if batch_size is None else batch_size
        num_workers = os.cpu_count() if num_workers is None else num_workers
        # I need to know how much I want to pad
        max_num_of_labels = dataset.get_max_num_of_labels(dataset.root)
        # let's get the first one so we can infer the shape
        image = dataset[0].image
        data: TensorDict = ObjectDetectionData(
            image=MemmapTensor(
                dataset_size,
                *image.shape[1:],
                dtype=torch.uint8,
                filename=str(self.dst) + "/image.memmap",
            ),
            bboxes=MemmapTensor(
                dataset_size,
                max_num_of_labels,
                4,
                dtype=torch.float32,
                filename=str(self.dst) + "/bboxes.memmap",
            ),
            labels=MemmapTensor(
                dataset_size,
                max_num_of_labels,
                dtype=torch.int64,
                filename=str(self.dst) + "/labels.memmap",
            ),
            images_sizes=MemmapTensor(
                dataset_size,
                2,
                dtype=torch.uint8,
                filename=str(self.dst) + "/images_sizes.memmap",
            ),
            batch_size=[dataset_size],
        )
        data.memmap_(prefix=self.dst)
        logger.debug(data)

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=os.cpu_count(),
            collate_fn=lambda x: x,
        )
        pbar = tqdm(total=dataset_size)

        for batch_idx, batch in enumerate(data_loader):
            batch_offset = batch_idx * batch_size
            batch: List[ObjectDetectionData] = batch
            for batch_data_idx, batch_data in enumerate(batch):
                # now I need to pad the bboxes
                padded_bboxes = torch.zeros(
                    (batch_data.shape[0], max_num_of_labels, 4),
                    dtype=batch_data.bboxes.dtype,
                )
                padded_bboxes[..., : padded_bboxes.shape[1], ...] = padded_bboxes
                batch_data.bboxes = padded_bboxes
                # and the labels!
                batch_data.labels = F.pad(
                    batch_data.labels,
                    (0, max_num_of_labels - batch_data.labels.shape[-1]),
                    value=0,
                )
                data[
                    batch_offset + batch_data_idx : batch_offset + batch_data_idx + 1
                ] = batch_data
                pbar.update(1)

        return data

    def _load(self) -> ObjectDetectionData:
        td = TensorDict.load_memmap(self.dst)
        logger.info(
            f"💾 Found a non empty folder at {self.dst}, loading dataset from there."
        )
        return ObjectDetectionData(**td, batch_size=td.batch_size)

    def build(
        self,
        dataset: YOLODataset,
        batch_size: int,
        num_workers: Optional[int] = None,
        overwrite_if_exists: bool = False,
    ) -> ObjectDetectionData:
        dataset_already_exists = self.dst.exists()
        if dataset_already_exists and not overwrite_if_exists:
            return self._load()
        return self._build(dataset, batch_size, num_workers)
