import math
from typing import Any, Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.nn.common import StackedConv2dLayers
from src.types import Features

class ScaleLayer(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_classes: int,
        depth: int = 2,
        features_stride: List[int] = (4, 8, 16, 32),
        prior_prob: float = 0.01,
    ):
        """
        Copied and adapter from OneNet: https://github.com/PeizeSun/OneNet/blob/main/projects/OneNet/onenet/head.py

        Args:
            in_channels (int): _description_
            channels (int): _description_
            num_classes (int): _description_
            depth (int, optional): _description_. Defaults to 2.
            features_stride (List[int], optional): _description_. Defaults to (4, 8, 16, 32).
            prior_prob (float, optional): _description_. Defaults to 0.01.
        """
        super().__init__()
        self.features_stride = features_stride
        self.scales = nn.ModuleList(
            [ScaleLayer(init_value=1.0) for _ in features_stride]
        )
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.class_prediction_branch = StackedConv2dLayers(in_channels, channels, depth)
        self.regression_prediction_branch = StackedConv2dLayers(
            in_channels, channels, depth
        )

        self.class_predictor = nn.Conv2d(
            channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bboxes_predictor = nn.Conv2d(
            channels, 4, kernel_size=3, stride=1, padding=1
        )

    #     self.init_weights(prior_prob)

    # def init_weights(self):
    #     # init all parameters.
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    #     # initialize the bias for focal loss.
    #     nn.init.constant_(self.cls_score.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, features: Features) -> Tuple[Tensor]:
        """

        Args:
            features (List[Tensor] of shape `(batch_size, channels, height, width)`): List of features the head uses

        Returns:
            Tuple[Tensor]: Tuple of Tensors of shape (`batch_size, num_queries, num_classes`) and (`batch_size, num_queries, 4)` representing the class logits and the predicted bboxes respectively. The predicted bboxes are explicit xyxy coordinates with respect of the original image
        """
        class_logits_all: List[Tensor] = []
        bboxes_predictions_all: List[Tensor] = []
        batch_size = features[0].shape[0]

        for feature, feature_stride, scale in zip(
            features, self.features_stride, self.scales
        ):
            # classes
            class_features = self.class_prediction_branch(feature)
            class_logits = self.class_predictor(class_features).view(
                batch_size, self.num_classes, -1
            )
            # bboxes
            regression_features = self.regression_prediction_branch(feature)
            locations_on_grid = self.get_locations_on_grid(feature, feature_stride)[
                None
            ]
            # bboxes here are center points for each cell, so the coodinares are (cx,cy,h,w)
            bboxes_predictions = self.bboxes_predictor(regression_features)
            # rescale bboxes by a learnable parameter
            bboxes_predictions = scale(bboxes_predictions)
            # rescale bboxes based on the level stride and force them to be [0,1]
            bboxes_predictions = F.relu(bboxes_predictions) * feature_stride
            # now the use locations_on_grid to get back the bboxes location on the image
            bboxes_predictions = self.to_xyxy_bboxes(
                locations_on_grid, bboxes_predictions
            ).view(batch_size, 4, -1)

            class_logits_all.append(class_logits)
            bboxes_predictions_all.append(bboxes_predictions)

        class_logits_all = (
            torch.cat(class_logits_all, dim=-1).permute(0, 2, 1).contiguous()
        )
        bboxes_predictions_all = (
            torch.cat(bboxes_predictions_all, dim=-1).permute(0, 2, 1).contiguous()
        )

        return class_logits_all, bboxes_predictions_all

    def to_xyxy_bboxes(self, locations: Tensor, pred_ltrb: Tensor) -> Tensor:
        """
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W)
        """

        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[:, 0, :, :] = locations[:, 0, :, :] - pred_ltrb[:, 0, :, :]  # x1
        pred_boxes[:, 1, :, :] = locations[:, 1, :, :] - pred_ltrb[:, 1, :, :]  # y1
        pred_boxes[:, 2, :, :] = locations[:, 0, :, :] + pred_ltrb[:, 2, :, :]  # x2
        pred_boxes[:, 3, :, :] = locations[:, 1, :, :] + pred_ltrb[:, 3, :, :]  # y2

        return pred_boxes

    @torch.no_grad()
    def get_locations_on_grid(self, features: Tensor, stride: int) -> Tensor:
        """
        This code essentially computes the (x, y) coordinates of the center points of evenly spaced cells in a grid, given the height and width of the grid and the stride between the cells.

        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        dtype, device = features.dtype, features.device
        # if stride is 8
        # [0, 8, 16, 24 ...]
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=dtype, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=dtype, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations_on_grid = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        locations_on_grid = locations_on_grid.reshape(h, w, 2).permute(2, 0, 1)

        return locations_on_grid


if __name__ == "__main__":
    # layers = StackedConv2dLayers(32, 64)
    # print(layers)

    head = Head(256, channels=256, num_classes=80)
    outs = head(
        [
            torch.randn((1, 256, 80, 80)),
            torch.randn((1, 256, 40, 40)),
            torch.randn((1, 256, 20, 20)),
            torch.randn((1, 256, 10, 10)),
            torch.randn((1, 256, 5, 5)),
        ]
    )

    print(outs[0].shape, outs[1].shape)
