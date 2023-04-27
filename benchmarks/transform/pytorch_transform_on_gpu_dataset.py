from functools import reduce
from pathlib import Path
from torch import nn
import torch
from torch import Tensor
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform=None) -> None:
        super().__init__()
        self.root = root
        self.images_paths = list(root.glob("*.jpg"))
        self.transform = transform

    def __getitem__(self, i: int) -> Tensor:
        image = read_image(str(self.images_paths[i]))
        if self.transform is not None:
            image = self.transform(image.float().div_(255).unsqueeze(0))[0]
        return image

    def __len__(self):
        return len(self.images_paths)


class Resize(nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, size=self.size)


class RandomFlip(nn.Module):
    def __init__(self, p: float = 0.5, dim: int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        flip_mask =  (
            torch.zeros(
                x.shape[0], 1, 1, 1, device=x.device, dtype=torch.bool
            ).bernoulli_(self.p)
        )
        x_flipped = x.flip(self.dim)
        x.mul_(~flip_mask)
        x.addcmul_(flip_mask, x_flipped)
        return x


class Normalize(nn.Module):
    def __init__(
        self, mean, std, device: str = "str", dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.mean, self.std = (
            torch.tensor(mean, device=device, dtype=dtype)[None, ..., None, None],
            torch.tensor(std, device=device, dtype=dtype)[None, ..., None, None],
        )

    def forward(self, x: Tensor) -> Tensor:
        x.sub_(self.mean).div_(self.std)
        return x


class Compose(nn.Module):
    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = transforms

    def forward(self, x: Tensor) -> Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


def get_benchmark_func(
    root: Path,
    device: str = "cpu",
    size: Tuple[int, int] = (640, 640),
    fp16: bool = False,
    compile: bool = False,
    **kwargs
):
    transform = Compose(
        [
            Resize(size),
            RandomFlip(dim=-1),
            RandomFlip(dim=-2),
            Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                device=device,
                dtype=torch.float16 if fp16 else torch.float32,
            ),
        ]
    )

    if compile:
        print("[INFO] compiling transform")
        transform = torch.compile(transform)

    ds = ImageDataset(root)
    dl = DataLoader(ds, **kwargs, pin_memory=device == "cuda")

    def _inner():
        for batch in tqdm(dl):
            batch = (
                batch.to(device).to(torch.float16 if fp16 else torch.float32).div_(255)
            )
            batch = transform(batch)
            yield batch

    return _inner


if __name__ == "__main__":

    benchmark_func = get_benchmark_func(
        Path("/home/zuppif/Documents/neatly/detector/datasets/train/images"),
        device="cuda",
        batch_size=32,
        num_workers=8,
    )

    for batch in benchmark_func():
        continue
