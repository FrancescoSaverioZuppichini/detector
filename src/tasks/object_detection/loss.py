from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision.ops.giou_loss import generalized_box_iou_loss

from ..matcher import Indices, Matcher


#  Copied and adapted from https://github.com/PeizeSun/OneNet/blob/19fa127c7c5896b99744458a92bf87e95c03ddad/projects/OneNet/onenet/loss.py
# [TODO]
# 1) decide exactly how to pass the inputs in forward, for now I went explicit
# 2) computer the num_bboxes for normalisation, I guess I could also just do "mean"
# 3) fix a lot of variables names to make it easier
class OneNetLoss(nn.Module):
    """
    This class computes the loss for OneNet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes: int, matcher: Matcher):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        # self.class_loss_weight = class_loss_weight
        # self.bbox_regression_loss_weight = bbox_regression_loss_weight
        # self.bbox_location_loss_weight = bbox_location_loss_weight
        # self.losses = losses
        self.focal_loss_alpha = 0.1
        self.focal_loss_gamma = 0.2

    def get_class_loss(self, class_logits, class_labels, indices: Indices) -> Tensor:
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        idx = self._get_src_permutation_idx(indices)
        # here we find out the classs ids of the matched bboxes
        target_classes_o = torch.cat(
            [class_labels[i, col] for i, (_, col) in enumerate(indices)]
        )
        print(class_labels)
        print(target_classes_o, "target_classes_o")

        # we need to create a tensor of shape `(batch_size, num_preds, num_classes)` where we place 1 in the correct class id in the last dimension (so one hot)
        target_classes = torch.full(
            class_logits.shape[:2],
            fill_value=self.num_classes,
            dtype=torch.int64,
            device=class_logits.device,
        )
        # now we place in the correct position (w.r. to predicted bboxes) the matched classes
        target_classes[idx] = target_classes_o
        # we flat the batch dim, to obtain `(num_preds)`
        class_logits = class_logits.flatten(0, 1)
        # we also flat the target_classes to correct find the position of the class labels
        target_classes = target_classes.flatten(0, 1)
        # now we find the position of the matched target labels, note that `num_classes` will be always greater than the actually number of classes (we start to count from 0)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        # labels have shape `(num_preds, num_classes)`, we need to one hot encode
        labels = torch.zeros_like(class_logits)
        # here we select the correct pred and set the coresponding class position to 1 (one hot encoding)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # compute focal loss
        class_loss = sigmoid_focal_loss(
            class_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        return class_loss

    def get_boxes_losses(
        self,
        boxes_preds: Tensor,
        boxes_labels: Tensor,
        indices: Indices,
        image_size: Tensor,
    ):
        idx = self._get_src_permutation_idx(indices)

        boxes_preds = boxes_preds[idx]
        target_boxes = torch.cat(
            [el[i] for el, (_, i) in zip(boxes_labels, indices)], dim=0
        )

        loss_giou = generalized_box_iou_loss(boxes_preds, target_boxes, reduction="sum")
        # [NOTE] hacky using only the first one
        loss_bbox = (
            F.l1_loss(boxes_preds, target_boxes, reduction="sum") / image_size[0]
        )

        return loss_giou, loss_bbox

    def _get_src_permutation_idx(self, indices: Indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: Indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(
        self,
        class_logits: Tensor,
        boxes_preds: Tensor,
        class_labels: Tensor,
        boxes_labels: Tensor,
        image_size: Tensor,
        mask_labels: Optional[Tensor] = None,
    ):
        """_summary_

        Args:
            class_logits (Tensor, of shape `(batch_size, num_queries, num_classes)`): _description_
            boxes_preds (Tensor of shape `(batch_size, num_queries, 4)`): The predicted bboxes, normalized by image size so between [0,1]
            class_labels (Tensor of shape `(batch_size, num_targets, num_classes)`): The target label, one class_id per bbox
            boxes_labels (Tensor of shape `(batch_size, num_targets, 4)`): The target boxes_labels, normalized by image size so between [0,1]
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(
            class_logits.sigmoid(), boxes_preds, class_labels, boxes_labels, mask_labels
        )
        # [TODO] we are missing the num_boxes here
        num_boxes = 1

        class_loss = self.get_class_loss(class_logits, class_labels, indices)
        bbox_location, bbox_regression_loss = self.get_boxes_losses(
            boxes_preds, boxes_labels, indices, image_size
        )

        # Compute all the requested losses
        losses = {
            "class_loss": class_loss,
            "bbox_location": bbox_location,
            "bbox_regression_loss": bbox_regression_loss,
        }

        return losses
