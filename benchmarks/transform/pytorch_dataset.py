from functools import reduce
from pathlib import Path

import torch
from torch import Tensor
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform=None) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.images_paths = list(root.glob("*.jpg"))

    def __getitem__(self, i: int) -> Tensor:
        image = read_image(str(self.images_paths[i])).float() / 255
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images_paths)


def get_benchmark_func(
    root: Path,
    device: str = "cpu",
    size: Tuple[int, int] = (640, 640),
    fp16: bool = False,
    compile: bool = False,
    **kwargs,
):
    transform = T.Compose(
        [
            T.Resize(size, antialias=False),
            T.RandomHorizontalFlip(.5),
            T.RandomVerticalFlip(.5),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ds = ImageDataset(root, transform)
    dl = DataLoader(ds, **kwargs, pin_memory=device == "cuda")

    def _inner():
        for batch in tqdm(dl):
            batch = batch.to(device)
            yield batch

    return _inner


if __name__ == "__main__":
    # ds = ImageDataset(
    #     Path("/home/zuppif/Documents/neatly/detector/datasets/train/images")
    # )
    # print(ds[0].shape)

    benchmark_func = get_benchmark_func(
        Path("/home/zuppif/Documents/neatly/detector/datasets/train/images"),
        device="cuda",
        batch_size=32,
        num_workers=4,
    )
    for batch in benchmark_func():
        # print(batch.shape)
        continue
