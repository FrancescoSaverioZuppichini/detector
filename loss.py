from torch import nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision.ops.boxes import generalized_box_iou
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import relu


class MinCostMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        """Creates the matcher

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
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    def forward(self, class_labels_logits, boxes_logits, class_labels, boxes):
        # boxes has to be xyxy and in pixels

        boxes_logits = boxes_logits.permute(0, 2, 1)  # [batch_size, num_queries, 4]
        class_labels_logits = class_labels_logits.permute(
            0, 2, 1
        )  # [batch_size, num_queries, num_classes]

        indices = []

        for i in range(class_labels.shape[0]):
            curr_class_labels_logits = class_labels_logits[i]  # is out_prob
            curr_boxes_logits = boxes_logits[i]  # is curr_boxes_logits
            curr_class_labels = class_labels[i]
            curr_boxes = boxes[i]  # is tgt_bbox
            if curr_class_labels.shape[0] == 0:
                indices.append((torch.as_tensor([]), torch.as_tensor([])))
                continue

            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (
                (1 - alpha)
                * (curr_class_labels_logits**gamma)
                * (-(1 - curr_class_labels_logits + 1e-8).log())
            )
            pos_cost_class = (
                alpha
                * ((1 - curr_class_labels_logits) ** gamma)
                * (-(curr_class_labels_logits + 1e-8).log())
            )
            cost_class = (
                pos_cost_class[:, curr_class_labels]
                - neg_cost_class[:, curr_class_labels]
            )

            # [NOTE] we need to normalize the boxes between 0, 1 - omitted here
            # Compute the L1 cost between boxes
            # image_size_out = targets[i]["image_size_xyxy"].unsqueeze(0).repeat(hw, 1)
            # image_size_tgt = targets[i]["image_size_xyxy_tgt"]
            # need to divide cuz we predict numbers!
            # curr_boxes_logits_ = curr_boxes_logits / image_size_out
            # tgt_bbox_ = tgt_bbox / image_size_tgt

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(curr_boxes_logits, curr_boxes, p=1)
            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(curr_boxes_logits, curr_boxes)
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )

            src_ind, tgt_ind = linear_sum_assignment(C.cpu())
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
        class_labels_logits=torch.randn((1, 80, 8525)),
        boxes_logits=relu(torch.randn((1, 4, 8525))),
        class_labels=torch.tensor([[1]]),
        boxes=torch.tensor([[[0.1, 0.1, 0.3, 0.3]]]),
    )
    print(res)
