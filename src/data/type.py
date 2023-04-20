from __future__ import annotations

import torch
from tensordict.prototype import tensorclass
import matplotlib.pyplot as plt
import numpy as np

@tensorclass
class ObjectDetectionData:
    image: torch.Tensor
    # format x1,y1,x2,y2
    bboxes: torch.Tensor
    labels: torch.Tensor
    images_sizes: torch.Tensor
    
    def show(self):
        data = self
        images = data.image.detach().cpu().numpy()
        bboxes = data.bboxes.detach().cpu().numpy()
        labels = data.labels.detach().cpu().numpy()
        batch_size = images.shape[0]

        if batch_size == 1:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            axs = np.array([axs])  # Convert to numpy array for indexing
        else:
            fig, axs = plt.subplots(nrows=batch_size, ncols=1, figsize=(8, 8 * batch_size))

        for i in range(batch_size):
            image = np.transpose(images[i], (1, 2, 0))
            axs[i].imshow(image)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            for bbox, label in zip(bboxes[i], labels[i]):
                if label == 0:
                    continue  # Ignore padded boxes
                bbox = bbox.tolist()
                x1, y1, x2, y2 = bbox

                color = plt.cm.tab20(label % 20)
                rect = plt.Rectangle((x1, y1),
                                    x2 - x1, y2 - y1,
                                    fill=False, edgecolor=color, linewidth=2.5)
                axs[i].add_patch(rect)

        plt.tight_layout()
        plt.show()
