from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict.prototype import tensorclass


@tensorclass
class ObjectDetectionData:
    images: torch.Tensor
    # format x1,y1,x2,y2
    bboxes: torch.Tensor
    classes: torch.Tensor
    images_sizes: torch.Tensor
    mask_labels: Optional[torch.Tensor] = None

    def show(self):
        data = self
        images = data.images.detach().cpu().numpy()
        bboxes = data.bboxes.detach().cpu().numpy()
        classes = data.classes.detach().cpu().numpy()
        batch_size = images.shape[0]

        if batch_size == 1:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            axs = np.array([axs])  # Convert to numpy array for indexing
        else:
            fig, axs = plt.subplots(
                nrows=batch_size, ncols=1, figsize=(8, 8 * batch_size)
            )

        for i in range(batch_size):
            images = np.transpose(images[i], (1, 2, 0))
            axs[i].imshow(images)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            for bbox, label in zip(bboxes[i], classes[i]):
                if label == 0:
                    continue  # Ignore padded boxes
                bbox = bbox.tolist()
                x1, y1, x2, y2 = bbox

                color = plt.cm.tab20(label % 20)
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.5,
                )
                axs[i].add_patch(rect)

        plt.tight_layout()
        plt.show()
