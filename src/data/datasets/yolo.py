from functools import reduce
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.data.type import ObjectDetectionData


class YOLODataset(Dataset):
    def __init__(self, root: Path, image_format: str = "jpg"):
        super().__init__()
        self.root = root
        self.image_format = image_format
        self.file_paths = list(self.get_file_paths(root, image_format))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> ObjectDetectionData:
        file_path = self.file_paths[idx]
        image, labels = self.get_image_and_labels(file_path)
        class_labels, bboxes_labels = labels[..., 0], labels[..., 1:]

        data = ObjectDetectionData(
            image[None, ...],
            bboxes_labels[None, ...],
            class_labels[None, ...],
            torch.tensor([image.shape[1:]], dtype=torch.uint8),
            batch_size=[1],
        )
        return data

    def get_labels(self, file_path: Path) -> torch.Tensor:
        label_file_path = file_path.parent.parent / "labels" / f"{file_path.stem}.txt"
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
        image = read_image(str(file_path))
        return image, labels

    def get_file_paths(self, root: Path, image_format: str = "jpg") -> list[Path]:
        return root.glob(f"**/*.{image_format}")

    @staticmethod
    def get_num_of_labels_in_a_file(file_path: Path):
        with file_path.open("r") as f:
            content = f.read()
            is_empty = content == ""
            num_of_labels_in_a_file = 0 if is_empty else len(content.split("\n"))
            return num_of_labels_in_a_file

    @staticmethod
    def get_num_of_labels(src: Path) -> int:
        num_of_labels = reduce(
            lambda x, y: x + y,
            map(
                YOLODataset.get_num_of_labels_in_a_file, (src / "labels").glob("*.txt")
            ),
        )
        return num_of_labels


if __name__ == "__main__":
    ds = YOLODataset(Path("/home/zuppif/Documents/neatly/detector/datasets/train"))
    print(ds[0])
