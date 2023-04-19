from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .type import ObjectDetectionData


class RandomHFlip(nn.Module):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, data: ObjectDetectionData) -> ObjectDetectionData:
        idx = self.generate_flip_mask(data.image)
        x_flipped = self.flip_image(data.image, idx)
        bboxes_flipped = self.flip_bboxes(data.bboxes, data.image, idx)
        return ObjectDetectionData(
            image=x_flipped,
            bboxes=bboxes_flipped,
            labels=data.labels,
            images_sizes=data.images_sizes,
            batch_size=data.batch_size,
        )

    def generate_flip_mask(self, x: torch.Tensor):
        idx = (
            torch.zeros(
                x.shape[0], 1, 1, 1, device=x.device, dtype=torch.bool
            ).bernoulli_(self.prob)
            # .expand_as(x)
        )
        return idx

    def flip_image(self, x: torch.Tensor, idx: torch.Tensor):
        # [NOTE] should change param from `idx` to flip_mask
        # x.masked_fill(idx, 0.0) -> set the image we want to flip to zero
        # x.masked_fill(~idx, 0.0) -> set the image we don't want to flip to zero, and we flip the other
        # when we sum
        x_flipped = x.masked_fill(idx, 0.0).add_(x.masked_fill(~idx, 0.0).flip(-1))
        return x_flipped

    def flip_bboxes(self, bboxes: torch.Tensor, x: torch.Tensor, idx: torch.Tensor):
        # here we basically clone the bboxes, we swap all of them and replace the original tensor with the swapped one based on the idx
        bboxes_flipped = bboxes.clone()
        bboxes_flipped[..., [0, 2]] = x.shape[-1] - bboxes[..., [2, 0]]
        bboxes.masked_scatter_(idx[..., 0, 0, 0].unsqueeze(-1), bboxes_flipped)
        return bboxes


class RandomCrop(nn.Module):
    def __init__(self, w, h, keep_aspect_ratio=False):
        super(RandomCrop, self).__init__()
        self.w = w
        self.h = h
        self.keep_aspect_ratio = keep_aspect_ratio

    def forward(self, data: ObjectDetectionData) -> ObjectDetectionData:
        index0, index1 = self.generate_crop_indices(data.image)
        data.image = self.crop_image(data.image, index0, index1)
        data.bboxes, data.labels = self.crop_bboxes(
            data.bboxes, data.labels, index0, index1
        )
        data.images_sizes = (
            torch.tensor([self.h, self.w], device=data.images_sizes.device)
            .unsqueeze(0)
            .expand_as(data.images_sizes)
        )
        return data

    def generate_crop_indices(self, x: torch.Tensor) -> tuple:
        batch = x.shape[:-3]
        index0 = torch.randint(x.shape[-2] - self.h, (*batch, 1), device=x.device)
        index0 = index0 + torch.arange(self.h, device=x.device)
        index0 = (
            index0.unsqueeze(1).unsqueeze(-1).expand(*batch, 3, self.h, x.shape[-1])
        )
        index1 = torch.randint(x.shape[-1] - self.w, (*batch, 1), device=x.device)
        index1 = index1 + torch.arange(self.w, device=x.device)
        index1 = index1.unsqueeze(1).unsqueeze(-2).expand(*batch, 3, self.h, self.w)

        return index0, index1

    def crop_image(
        self, x: torch.Tensor, index0: torch.Tensor, index1: torch.Tensor
    ) -> torch.Tensor:
        x_cropped = x.gather(-2, index0).gather_(-1, index1)

        if self.keep_aspect_ratio:
            padded_h, padded_w = self.calculate_padded_size(
                x.shape[-2], x.shape[-1], self.h, self.w
            )
            pad_top = (padded_h - self.h) // 2
            pad_bottom = padded_h - self.h - pad_top
            pad_left = (padded_w - self.w) // 2
            pad_right = padded_w - self.w - pad_left

            x_cropped = F.pad(
                x_cropped,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )

        return x_cropped

    def calculate_padded_size(self, orig_h, orig_w, target_h, target_w):
        orig_aspect_ratio = orig_h / orig_w
        target_aspect_ratio = target_h / target_w

        if orig_aspect_ratio > target_aspect_ratio:
            padded_h = target_h
            padded_w = int(target_h / orig_aspect_ratio)
        else:
            padded_h = int(target_w * orig_aspect_ratio)
            padded_w = target_w

        return padded_h, padded_w

    def crop_bboxes(
        self,
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        index0: torch.Tensor,
        index1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_crop_start = index0[..., 0].unsqueeze(-1)
        x_crop_start = index1[..., 0].unsqueeze(-1)
        bboxes_cropped = bboxes.clone()

        bboxes_cropped[..., [0, 2]] -= x_crop_start
        bboxes_cropped[..., [1, 3]] -= y_crop_start

        # Clamp the coordinates to the crop area
        bboxes_cropped[..., [0, 2]] = torch.clamp(
            bboxes_cropped[..., [0, 2]], 0, self.w
        )
        bboxes_cropped[..., [1, 3]] = torch.clamp(
            bboxes_cropped[..., [1, 3]], 0, self.h
        )

        # Compute the bounding box area after cropping
        cropped_area = (bboxes_cropped[..., 2] - bboxes_cropped[..., 0]) * (
            bboxes_cropped[..., 3] - bboxes_cropped[..., 1]
        )

        # Compute the original bounding box area
        original_area = (bboxes[..., 2] - bboxes[..., 0]) * (
            bboxes[..., 3] - bboxes[..., 1]
        )

        # Compute the ratio of cropped area to the original area
        area_ratio = cropped_area / original_area

        # Set the labels to zero (background) for bounding boxes with area ratio below a threshold (e.g., 0.5)
        labels_cropped = labels.clone()
        labels_cropped[area_ratio < 0.5] = 0
        # # [NOTE] this one removes the labels
        # # Alternatively, remove the bounding boxes with area ratio below the threshold
        # valid_indices = area_ratio >= 0.5
        # bboxes_cropped = bboxes_cropped[valid_indices]
        # labels_cropped = labels_cropped[valid_indices]

        return bboxes_cropped, labels_cropped


class Resize(nn.Module):
    def __init__(self, size: Tuple[int, int], keep_aspect_ratio: bool = False):
        super().__init__()
        self.size = size
        self.keep_aspect_ratio = keep_aspect_ratio

    def compute_scale_factors(self, images_sizes: torch.Tensor) -> torch.Tensor:
        if self.keep_aspect_ratio:
            scale_factors = (
                torch.tensor(self.size, device=images_sizes.device).view(1, -1)
                / images_sizes
            )
            return torch.min(scale_factors, dim=-1, keepdim=True)[0].expand_as(
                scale_factors
            )
        else:
            return (
                torch.tensor(self.size, device=images_sizes.device).view(1, -1)
                / images_sizes
            )

    def resize_image(self, image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            image.unsqueeze(0), size, mode="bilinear", align_corners=False
        ).squeeze(0)

    def resize_bboxes(
        self, bboxes: torch.Tensor, scale_factors: torch.Tensor
    ) -> torch.Tensor:
        bboxes = bboxes * scale_factors.unsqueeze(-1)
        return bboxes

    def shift_bboxes(self, bboxes: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        bboxes[..., [0, 2]] += padding[..., 0].unsqueeze(-1)
        bboxes[..., [1, 3]] += padding[..., 1].unsqueeze(-1)
        return bboxes

    def forward(self, data: ObjectDetectionData) -> ObjectDetectionData:
        scale_factors = self.compute_scale_factors(data.images_sizes)
        if self.keep_aspect_ratio:
            target_sizes = (data.images_sizes * scale_factors).long()
            padding = (
                (
                    torch.tensor(self.size, device=data.images_sizes.device)
                    - target_sizes
                )
                / 2
            ).long()
            data.image = torch.stack(
                [
                    self.resize_image(image, tuple(target_size))
                    for image, target_size in zip(data.image, target_sizes)
                ]
            )
            data.bboxes = self.resize_bboxes(data.bboxes, scale_factors)
            data.bboxes = self.shift_bboxes(data.bboxes, padding)
            data.images_sizes = torch.tensor(
                self.size, device=data.images_sizes.device
            ).expand_as(data.images_sizes)
        else:
            data.image = torch.stack(
                [self.resize_image(image, self.size) for image in data.image]
            )
            data.bboxes = self.resize_bboxes(data.bboxes, scale_factors)
            data.images_sizes = torch.tensor(
                self.size, device=data.images_sizes.device
            ).expand_as(data.images_sizes)

        return data


class Augmentation(nn.Module):
    def __init__(self, generator: torch.Generator = None):
        self._generator = generator

    def set_generator(self, generator: torch.Generator = None):
        self._generator.set_state(generator.get_state())


class SequentialAugmentation(Augmentation):
    def __init__(
        self, augmentations: List[Augmentation], generator: torch.Generator = None
    ):
        super().__init__(generator=generator)
        self.augmentations = augmentations

    def _add_generator_to_augmentations(self):
        for augmentation in self.augmentations:
            augmentation.set_generator(self.generator)

    def forward(self, data: ObjectDetectionData) -> ObjectDetectionData:
        for augmentation in self.augmentations:
            data = augmentation(data)
        return data
