from enum import Enum
from functools import reduce
from pathlib import Path

import torch
from torchvision.io import read_image


def get_labels(file_path: Path) -> torch.Tensor:
    label_file_path = file_path.parent.parent / "labels" / f"{file_path.stem}.txt"
    with label_file_path.open("r") as f:
        # reading all labels in one go
        # ['label x y h w', ....]
        content = f.read()
        is_empty = content == ""
        if is_empty:
            return torch.empty(0, 5)
        labels_raw = content.split("\n")
        labels_raw = [row.split(" ") for row in labels_raw]
        labels = torch.tensor([[float(el) for el in row] for row in labels_raw])

    return labels


def get_image_and_labels(file_path: Path) -> tuple[torch.Tensor, torch.IntTensor]:
    labels = get_labels(file_path)
    image = read_image(str(file_path))
    return image, labels


def get_file_paths(src: Path, fmt: str = "jpg") -> list[Path]:
    return src.glob(f"**/*.{fmt}")


def get_num_of_labels_in_a_file(file_path: Path):
    with file_path.open("r") as f:
        content = f.read()
        is_empty = content == ""
        num_of_labels_in_a_file = 0 if is_empty else len(content.split("\n"))
        return num_of_labels_in_a_file


def get_num_of_labels(src: Path) -> int:
    num_of_labels = reduce(
        lambda x, y: x + y,
        map(get_num_of_labels_in_a_file, (src / "labels").glob("*.txt")),
    )
    return num_of_labels


def get_max_num_of_labels_per_image(src: Path) -> int:
    num_of_labels = max(
        map(get_num_of_labels_in_a_file, (src / "labels").glob("*.txt"))
    )
    return num_of_labels
