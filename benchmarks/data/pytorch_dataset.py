from functools import reduce
from pathlib import Path

import torch
from torch import Tensor
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from tqdm import tqdm
import torch.nn.functional as F


class YOLODataset(Dataset):
    def __init__(self, root: Path, image_format: str = "jpg", padding: bool = False):
        super().__init__()
        self.root = root
        self.image_format = image_format
        self.file_paths = list(self.get_file_paths(root, image_format))
        self.padding = padding
        if self.padding:
            self.max_num_of_labels = self.get_max_num_of_labels(root)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        file_path = self.file_paths[idx]
        image, labels = self.get_image_and_labels(file_path)
        class_labels, bboxes_labels = labels[..., 0], labels[..., 1:]

        if self.padding:
            padded_bboxes = torch.zeros(
                (self.max_num_of_labels, 4), dtype=bboxes_labels.dtype
            )
            padded_bboxes[..., : padded_bboxes.shape[1], ...] = padded_bboxes
            bboxes_labels = padded_bboxes
            class_labels = F.pad(
                class_labels,
                (0, self.max_num_of_labels - class_labels.shape[-1]),
                value=0,
            )

        return image, class_labels, bboxes_labels

    def get_labels(self, file_path: Path) -> torch.Tensor:
        label_file_path = file_path
        with label_file_path.open("r") as f:
            content = f.read()
            is_empty = content == ""
            if is_empty:
                return torch.empty(0, 5)
            labels_raw = content.split("\n")
            labels_raw = [row.split(" ") for row in labels_raw]
            labels = torch.tensor([[float(el) for el in row] for row in labels_raw])
        return labels

    def get_image_and_labels(
        self, file_path: Path
    ) -> tuple[torch.Tensor, torch.IntTensor]:
        labels = self.get_labels(file_path)
        image_file_path = (
            file_path.parent.parent / "images" / f"{file_path.stem}.{self.image_format}"
        )
        image = read_image(str(image_file_path))

        return image, labels

    def get_file_paths(self, root: Path, image_format: str = "jpg") -> list[Path]:
        return root.glob(f"**/*.txt")

    @staticmethod
    def get_num_of_labels_in_a_file(file_path: Path):
        with file_path.open("r") as f:
            content = f.read()
            is_empty = content == ""
            num_of_labels_in_a_file = 0 if is_empty else len(content.split("\n"))
            return num_of_labels_in_a_file

    @staticmethod
    def get_max_num_of_labels(src: Path) -> int:
        max_num_of_labels = max(
            map(
                YOLODataset.get_num_of_labels_in_a_file, (src / "labels").glob("*.txt")
            ),
        )
        return max_num_of_labels


def get_benchmark_func(root: Path, num_iter: int = 20, device: str = None, **kwargs):
    ds = YOLODataset(root, padding=True)
    dl = DataLoader(ds, **kwargs, pin_memory=device == "cuda")

    def _inner():
        for _ in range(num_iter):
            for batch in tqdm(dl):
                if device == "cuda":
                    image = batch[0].to("cuda")
                    class_labels = batch[1].to("cuda")
                    bboxes_labels = batch[2].to("cuda")
                yield

    return _inner
    # for (image, class_labels, bboxes_labels) in tqdm(dl):
    #     if device == "cuda":
    #         image = image.to("cuda")
    #         class_labels = class_labels.to("cuda")
    #         bboxes_labels = bboxes_labels.to("cuda")
    #     yield


if __name__ == "__main__":

    # scalene_profiler.stop()
    # ds = YOLODataset(Path("/home/zuppif/Documents/neatly/detector/datasets/train"))
    # print(ds[0])
    benchmark_func = get_benchmark_func(
        Path("/home/zuppif/Documents/neatly/detector/datasets/train"),
        # Path(
        #     "/home/zuppif/Documents/neatly/detector/benchmarks/data/datasets/lettuce-pallets-640/train"
        # ),
        num_workers=4,
    )

    # scalene_profiler.start()
    list(benchmark_func)
    # scalene_profiler.stop()
