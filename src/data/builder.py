from pathlib import Path

import torch
from tensordict import MemmapTensor, TensorDict

from .datasets.yolo import YOLODataset
from .type import ObjectDetectionData


class ObjectDetectionDatasetBuilder:
    def __init__(self, dst: Path):
        self.dst = dst

    def _maybe_create_dst(self):
        self.dst.mkdir(exist_ok=True, parents=True)

    def build(self, dataset: YOLODataset) -> ObjectDetectionData:
        self._maybe_create_dst()
        dataset_size = len(dataset)
        # I need to know how much I want to pad
        max_num_of_labels = dataset.get_max_num_of_labels_per_image(dataset.root)
        # Let's get the first one so we can infer the shape
        img, _ = dataset(0)
        data = ObjectDetectionData(
            image=MemmapTensor(
                dataset_size,
                *img.shape,
                dtype=torch.uint8,
            ),
            bboxes=MemmapTensor(dataset_size, max_num_of_labels, 4, dtype=torch.uint8),
            labels=MemmapTensor(dataset_size, 1, dtype=torch.float64),
            images_sizes=MemmapTensor(dataset_size, 2, dtype=torch.uint8),
            batch_size=[dataset_size],
        )


#  @classmethod
#     def from_dataset(cls, src: Path):
#         file_paths = list(get_file_paths(src))
#         img, _ = get_image_and_labels(file_paths[0])
#         num_images = len(file_paths)
#         max_num_of_labels = get_max_num_of_labels_per_image(src)
#         # N = num of images, M = bboxes per image -> NxMx...
#         data: TensorDict = cls(
#             images=MemmapTensor(
#                 num_images,
#                 *img.shape,
#                 dtype=torch.uint8,
#             ),
#             labels=MemmapTensor(num_images, max_num_of_labels, 5, dtype=torch.float32),
#             labels_offsets=MemmapTensor(num_images, dtype=torch.long),
#             # bboxes=MemmapTensor(num_images, dtype=torch.float32),
#             batch_size=[num_images],
#         )
#         data = data.memmap_()

#         # dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
#         i = 0

#         def wrapper_to_add_to_data(i: int):
#             image, labels = get_image_and_labels(file_paths[i])
#             num_labels = labels.shape[0] # Mx5 [class id, w, h, xc, yc]
#             padded_labels = torch.empty((max_num_of_labels, 5))
#             padded_labels[:num_labels] = labels
#             _batch = 1
#             pbar.update(_batch)
#             data[i : i + _batch] = cls(
#                 images=image.unsqueeze(0),
#                 labels=padded_labels.unsqueeze(0),
#                 labels_offsets=torch.tensor([num_labels], dtype=torch.long).unsqueeze(
#                     0
#                 ),
#                 batch_size=[_batch],
#             )

#         pipe = pl.thread.map(wrapper_to_add_to_data, range(len(file_paths)), workers=16, maxsize=8)

#         pbar = tqdm(total=len(file_paths))
#         _ = list(pipe)

#         return data
