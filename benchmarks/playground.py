import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tensordict import TensorDict

images = torch.randn((10, 3, 224, 224))
bboxes = torch.randn((10, 100, 4))
from pathlib import Path

ds = TensorDataset(images, bboxes)
from data.tensordict_dataset import get_benchmark_func

# class MyDataset(Dataset):
#     def __init__(self, size: int = 100) -> None:
#         super().__init__()
#         self.size = size

#     def __getitem__(self, index) -> TensorDict:
#         return { 'image':  torch.randn((1, 3, 224, 224)), 'bboxes':torch.randn((1, 100, 4))}

#     def __len__(self):
#         return self.size

# ds = MyDataset()
# dl = DataLoader(ds, batch_size=4)

for batch in get_benchmark_func(
    Path("/home/zuppif/Documents/neatly/detector/datasets/train"),
    dst=Path("/home/zuppif/Documents/neatly/detector/benchmarks/tmp"),
    batch_size=32,
):
    print(batch.labels.data)
# for batch in dl:
#     print(batch[1].shape, batch[1].shape)
