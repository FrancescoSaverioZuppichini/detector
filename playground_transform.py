import torch
from torch import nn
from typing import Tuple, List
from src.data.type import ObjectDetectionData
import torch.nn.functional as F
from src.data.builder import ObjectDetectionDatasetBuilder
from src.data.datasets.yolo import YOLODataset
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial, reduce
from src.utils import profile_on_cuda
from src.data.transform import Normalizer, Resize
from torchvision.ops.boxes import box_convert
from torchvision.io import read_image


image = read_image("grogu.jpg")
images = torch.stack([image, image.clone()], dim=0)
bboxes = torch.tensor([[[395, 140, 763, 633], [200, 100, 300, 300]], [[395, 140, 763, 633], [200, 100, 300, 300]]])
labels = torch.tensor([[[1],[2]], [[1],[2]]])
images_sizes = torch.tensor([image.shape[1:], image.shape[1:]])

data = ObjectDetectionData(images, bboxes, labels, images_sizes, batch_size=[images.shape[0]], device="cpu")
data.images_sizes


res = Resize((200,200), keep_aspect_ratio=True)
print(data.images_sizes)
print(data.bboxes)
data_tr = res(data.clone()) 
print(data_tr.bboxes)
(data_tr).show()

# images = torch.randn(2, 3, 200, 400)
# bboxes = torch.randn(2, 2, 4)
# labels = torch.randint(2, (2,2), dtype=torch.int64)
# images_sizes = torch.tensor([[200, 400], [200, 400]])

# data = ObjectDetectionData(images, bboxes, labels, images_sizes, batch_size=[2], device="cpu")


# class BoxConverter(nn.Module):
#     FORMATS = ['xyxy', 'xywh', 'cxcywh']
#     def __init__(self, in_fmt: str, out_fmt: str):
#         super().__init__()
#         if in_fmt not in self.FORMATS:
#             raise ValueError(f'in_fmt={in_fmt} not in {self.FORMATS}')
#         if out_fmt not in self.FORMATS:
#             raise ValueError(f'out_fmt={out_fmt} not in {self.FORMATS}')
#         self.in_fmt = in_fmt
#         self.out_fmt = out_fmt

#     def forward(self, data: ObjectDetectionData) -> ObjectDetectionData:
#         data.bboxes = box_convert(data.bboxes, self.in_fmt, self.out_fmt)
#         return data

# print(data.bboxes)
# tr = BoxConverter('xyxy', 'xywh')
# print(tr(data).bboxes)
#     # print(normalizer(data).image.dtype)
# # tr = Resize(size=(200,300), keep_aspect_ratio=True)
# # res = tr(data)
# # print(res)

# ds = YOLODataset(root=Path("/home/zuppif/Documents/neatly/detector/datasets/train"))
# builder = ObjectDetectionDatasetBuilder(dst=Path('memmap'))
# data = builder.build(ds, batch_size=64, overwrite_if_exists=True)
# print(data)

# def my_collate_fn(batch: ObjectDetectionData, device: torch.device):
#     # batch = batch.apply(lambda tensor: tensor.contiguous().pin_memory())
#     if device.type == "cuda":
#         batch = batch.to("cuda", non_blocking=True)
#     return batch


# def get_size_mb(tensor: torch.Tensor) -> float:
#     num_elements = tensor.numel()

#     # get the size in MB
#     size_in_bytes = num_elements * tensor.element_size()
#     size_in_mb = size_in_bytes / (1024 * 1024)

#     return size_in_mb



# print(f"Needed MB = {get_size_mb(data.image) + get_size_mb(data.bboxes)}")

# @profile_on_cuda
# def profile():
#     data.apply(lambda tensor: tensor.contiguous().pin_memory().to("cuda", non_blocking=True))
# dl = DataLoader(data, collate_fn=partial(my_collate_fn, device=torch.device("cuda")), batch_size=512)
# # data.apply(lambda _tensor: _tensor.contiguous())
# total = data.image.shape[0]

# for _ in range(10):
#     pbar = tqdm(total=total)
#     for data in dl:
#         pbar.update(data.image.shape[0])
#         continue
