from __future__ import annotations

import torch
from tensordict.prototype import tensorclass
from pathlib import Path
from pytorch_dataset import YOLODataset
import os
from tensordict import MemmapTensor, TensorDict
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
import torch.nn.functional as F
from functools import partial


@tensorclass
class ObjectDetectionData:
    image: torch.Tensor
    # format x1,y1,x2,y2
    bboxes: torch.Tensor
    labels: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset: YOLODataset, dst: Path):
        dst.mkdir(exist_ok=True, parents=True)
        dataset_size = len(dataset)

        if dst.exists():
            data = TensorDict.load_memmap(dst)
            data.memmap_(prefix=dst)

            return cls(**data, batch_size=[dataset_size])

        batch_size = 8
        # I need to know how much I want to pad
        max_num_of_labels = dataset.get_max_num_of_labels(dataset.root)
        # let's get the first one so we can infer the shape
        image = dataset[0][0]
        print(image.shape)
        data: TensorDict = ObjectDetectionData(
            image=MemmapTensor(
                dataset_size,
                *image.shape,
                dtype=torch.uint8,
                filename=str(dst) + "/image.memmap",
            ),
            bboxes=MemmapTensor(
                dataset_size,
                max_num_of_labels,
                4,
                dtype=torch.float32,
                filename=str(dst) + "/bboxes.memmap",
            ),
            labels=MemmapTensor(
                dataset_size,
                max_num_of_labels,
                dtype=torch.int64,
                filename=str(dst) + "/labels.memmap",
            ),
            batch_size=[dataset_size],
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=os.cpu_count(),
            collate_fn=lambda x: x,
        )
        print("Building")
        data.memmap_(prefix=dst)
        pbar = tqdm(total=dataset_size)

        for batch_idx, batch in enumerate(data_loader):
            batch_offset = batch_idx * batch_size
            for batch_data_idx, (image, class_labels, bboxes_labels) in enumerate(
                batch
            ):
                data[
                    batch_offset + batch_data_idx : batch_offset + batch_data_idx + 1
                ] = cls(
                    image.unsqueeze(0),
                    bboxes_labels.unsqueeze(0),
                    class_labels.unsqueeze(0),
                    batch_size=[1],
                )

                pbar.update(1)

        return data


def collate_fn(x, device: str = "cpu"):
    x = x.contiguous()
    if device == "cuda":
        x = x.apply(lambda x: x.as_tensor()).pin_memory()
    # else:
    #     out = x.apply(lambda x: x.as_tensor())
    return x


def get_benchmark_func(
    root: Path, dst: Path, num_iter: int = 20, device: str = None, **kwargs
):
    ds = YOLODataset(root, padding=True)
    data = ObjectDetectionData.from_dataset(ds, dst)
    dl = DataLoader(
        data,
        persistent_workers=False,
        **kwargs,
        # pin_memory=device == "cuda",
        collate_fn=partial(collate_fn, device=device),
    )

    def _inner():
        for _ in range(num_iter):
            for batch in tqdm(dl):
                batch = batch.to(device, non_blocking=True)
                yield batch

    return _inner
