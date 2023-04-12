from torch import nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch
from torch import Tensor
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
from torchvision.ops.giou_loss import generalized_box_iou_loss
from typing import Optional, Tuple, List
from matcher import Matcher, Indices


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
        # print(class_labels.shape)
        # print(list(indices))
        target_classes_o = torch.cat(
            [class_labels[col] for (_, col) in indices]
        )
        print(class_labels)
        print(target_classes_o, "target_classes_o")

        # we need to create a tensor of shape `(batch_size, num_predictions, num_classes)` where we place 1 in the correct class id in the last dimension (so one hot)
        target_classes = torch.full(
            class_logits.shape[:2],
            fill_value=self.num_classes,
            dtype=torch.int64,
            device=class_logits.device,
        )
        # then we add the right class id
        target_classes[idx] = target_classes_o
        # we flat the batch dim
        class_logits = class_logits.flatten(0, 1)
        # prepare one_hot encoding, we flat as well
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(class_logits)
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

    def get_boxes_losses(self, boxes_preds: Tensor, boxes_labels: Tensor, indices: Indices, image_size: Tensor):
        idx = self._get_src_permutation_idx(indices)

        boxes_preds = boxes_preds[idx]
        target_boxes = torch.cat(
            [el[i] for el, (_, i) in zip(boxes_labels, indices)], dim=0
        )

        loss_giou = generalized_box_iou_loss(
            boxes_preds, target_boxes, reduction="sum"
        )

        loss_bbox = F.l1_loss(boxes_preds, target_boxes, reduction="sum") / image_size

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
            boxes_preds, boxes_labels, indices
        )

        # Compute all the requested losses
        losses = {
            "class_loss": class_loss,
            "bbox_location": bbox_location,
            "bbox_regression_loss": bbox_regression_loss,
        }

        return losses


if __name__ == "__main__":
    from matcher import MinCostMatcher

    criterion = OneNetLoss(2, MinCostMatcher())
    torch.manual_seed(0)
    num_queries = 3
    num_classes = 2

    indices = [(torch.tensor([0, 1]), torch.tensor([1, 0], dtype=torch.int64))]
    class_logits = torch.tensor([[[ 1.5410, -0.2934],
         [-2.1788,  0.5684],
         [-1.0845, -1.3986]]])
    loss = criterion.get_class_loss(
        class_logits=class_logits,
        class_labels=torch.tensor([1, 0]),
        # left pred index, right label index
        indices=indices
    )
    # has to be 0.8772 to match original implementation, it's correct
    print(loss)
    boxes_preds=torch.tensor([[[10,10,50,50], [20,20,80,80], [10,10,50,50]]]).float()
    print(boxes_preds.shape)
    boxes_labels=torch.tensor([[[10,10,50,50], [20,20,80,80]]])
    print(boxes_labels.shape)
    loss = criterion.get_boxes_losses(
        boxes_preds=boxes_preds,
        boxes_labels=boxes_labels,
        # left pred index, right label index
        image_size=torch.tensor(640),
        indices=indices
    )
    print(loss)
    # has to be (tensor(1.8263), tensor(0.2500)), it's correct