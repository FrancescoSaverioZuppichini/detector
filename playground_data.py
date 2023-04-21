import sys
sys.path.append('/home/zuppif/Documents/neatly/detector/')
from src.data.datasets.yolo import YOLODataset
from pathlib import Path
from src.data.transform import SequentialAugmentation, RandomHFlip
import torch
from src.data.type import ObjectDetectionData

ds = YOLODataset(Path("/home/zuppif/Documents/neatly/detector/datasets/train"))
print(ds[0].image.shape)

x: ObjectDetectionData = RandomHFlip(1)(ds[0])
print(x.image.shape)
assert torch.equal(ds[0].image, x.image.flip(-1))


