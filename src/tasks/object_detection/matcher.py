from typing import List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops.boxes import generalized_box_iou
from torchvision.ops.focal_loss import sigmoid_focal_loss

Indices = List[Tuple[Tensor, Tensor]]


class Matcher(nn.Module):
    def forward(self, *args, **kwargs) -> Indices:
        raise NotImplementedError


class MinCostMatcher(Matcher):
    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        """Creates a min cost matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss_alpha = 0.1
        self.focal_loss_gamma = 0.3
        # [TODO] brutal assertion
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(
        self,
        class_preds: Tensor,
        boxes_preds: Tensor,
        class_labels: Tensor,
        boxes_labels: Tensor,
        mask_labels: Optional[Tensor] = None,
    ) -> Indices:
        """[TODO]

        Args:
            class_preds (Tensor, of shape `(batch_size, num_queries, num_classes)`): The predicted classes, between [0,1]
            boxes_preds (Tensor of shape `(batch_size, num_queries, 4)`): The predicted bboxes_labels, normalized by image size so between [0,1]
            class_labels (Tensor of shape `(batch_size, num_targets)`): The target label, one class_id per bbox
            boxes_labels (Tensor of shape `(batch_size, num_targets, 4)`): The target bboxes, normalized by image size so between [0,1],
            mask_labels (Tensor of shape `(batch_size, num_targets)`): Optional, a bool tensor indicating were padding was applied on the target


        Returns:
            List[Tuple[Tensor, Tensor]]: The list of matched indices
        """
        # boxes_labels has to be xyxy and in pixels
        boxes_preds = boxes_preds
        class_preds = class_preds

        indices = []

        should_remove_pad = mask_labels is not None
        # [TODO] we should be able to batch most of this computation
        for i in range(class_labels.shape[0]):
            curr_class_preds = class_preds[i]  # is out_prob
            curr_boxes_preds = boxes_preds[i]  # is curr_boxes_preds
            curr_class_labels = class_labels[i]
            curr_boxes_labels = boxes_labels[i]  # is tgt_bbox

            if should_remove_pad:
                curr_class_labels = curr_class_labels[mask_labels[i]]
                curr_boxes_labels = curr_boxes_labels[mask_labels[i]]

            if curr_class_labels.shape[0] == 0:
                indices.append((torch.as_tensor([]), torch.as_tensor([])))
                continue
            # [TODO] each subloss in its function
            # compute the classification cost
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (
                (1 - alpha)
                * (curr_class_preds**gamma)
                * (-(1 - curr_class_preds + 1e-8).log())
            )
            pos_cost_class = (
                alpha
                * ((1 - curr_class_preds) ** gamma)
                * (-(curr_class_preds + 1e-8).log())
            )
            cost_class = (
                pos_cost_class[:, curr_class_labels]
                - neg_cost_class[:, curr_class_labels]
            )
            # compute the L1 cost between bboxes
            cost_bbox = torch.cdist(curr_boxes_preds, curr_boxes_labels, p=1)
            # compute the giou cost between bboxes
            cost_giou = -generalized_box_iou(curr_boxes_preds, curr_boxes_labels)
            cost_matrix = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
            # we have to send it to cpu to use scipy
            # [TODO] any better method? Decouple this to a strategy
            src_ind, tgt_ind = linear_sum_assignment(cost_matrix.cpu())
            indices.append((src_ind, tgt_ind))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


if __name__ == "__main__":
    matcher = MinCostMatcher()
    res = matcher(
        class_preds=torch.tensor(
            [
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                ]
            ]
        ),
        boxes_preds=torch.tensor(
            [
                [
                    [0.2, 0.2, 0.5, 0.5],
                    [0.1, 0.1, 0.3, 0.3],
                ]
            ]
        ),
        class_labels=torch.tensor([[1]]),
        boxes_labels=torch.tensor([[[0.1, 0.1, 0.3, 0.3]]]),
    )
    print(res)

    res = matcher(
        class_preds=torch.tensor(
            [
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                ],
                [[0.1, 0.5], [0.1, 0.5]],
            ]
        ),
        boxes_preds=torch.tensor(
            [
                [
                    [0.2, 0.2, 0.5, 0.5],
                    [0.1, 0.1, 0.3, 0.3],
                ],
                [[0.2, 0.2, 0.5, 0.5], [0.2, 0.2, 0.5, 0.5]],
            ]
        ),
        class_labels=torch.tensor(
            [
                [1, 0],
                [0, 0],
            ]
        ),
        boxes_labels=torch.tensor(
            [
                [
                    [0.1, 0.1, 0.3, 0.3],
                    [0.2, 0.2, 0.4, 0.4],
                ],
                [[0.1, 0.1, 0.3, 0.3], [0, 0, 0, 0]],  # pad
            ]
        ),
        mask_labels=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
        # image_size=torch.Tensor([10])
    )
    print(res)
