from ffcv.writer import DatasetWriter

from data.pytorch_dataset import YOLODataset
from ffcv.fields import NDArrayField, FloatField
from ffcv.fields import RGBImageField, IntField, FloatField
from ffcv.fields.decoders import SimpleRGBImageDecoder, IntDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from ffcv.transforms import ToTensor
from pathlib import Path
import numpy as np

ds = YOLODataset(
    Path("/home/zuppif/Documents/neatly/detector/datasets/train"), padding=True
)
print([el.shape for el in ds[1]])

print(ds.max_num_of_labels)
writer = DatasetWriter(
    "dataset.beton",
    {
        "image": RGBImageField(),
        "label": NDArrayField(shape=(ds.max_num_of_labels, 1), dtype=np.dtype("float")),
        "bbox": NDArrayField(shape=(ds.max_num_of_labels, 4), dtype=np.dtype("float")),
    },
    num_workers=4,
)


writer.from_indexed_dataset(ds)

loader = Loader(
    "dataset.beton",
    batch_size=32,
    num_workers=8,
    order=OrderOption.RANDOM,
    pipelines={
        "image": [SimpleRGBImageDecoder(), ToTensor()],
        "bbox": [NDArrayDecoder(), ToTensor()],
        "label": [NDArrayDecoder(), ToTensor()],
    },
)


for batch in loader:
    print(batch[0].shape)
