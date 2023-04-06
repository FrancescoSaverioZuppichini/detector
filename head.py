from typing import Optional, List, Callable, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from functools import partial
from common import ConvNormGELULayer


class ScaleLayer(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class StackedConv2dLayers(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        conv_layer: Callable[[Any], nn.Module] = ConvNormGELULayer,
        *additional_layers
    ):
        super().__init__()
        self.layers = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=3),
            *[
                conv_layer(out_channels, out_channels, kernel_size=3)
                for _ in range(depth - 1)
            ],
            *additional_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        depth: int = 2,
        features_stride: List[int] = (8, 16, 32, 64, 128),
        num_classes: int = 80,
    ):
        super().__init__()
        num_levels = len(features_stride)
        self.features_stride = features_stride
        self.scales = nn.ModuleList(
            [ScaleLayer(init_value=1.0) for _ in range(num_levels)]
        )
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_classes = num_classes

        # [NOTE] we may need RELU in the last layer to force  0 - 1
        self.class_prediction_branch = StackedConv2dLayers(in_channels, channels, depth)
        self.regression_prediction_branch = StackedConv2dLayers(
            in_channels, channels, depth
        )

        # was cls_score
        self.class_predictor = nn.Conv2d(
            channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        # was ltrb_pred
        self.bboxes_predictor = nn.Conv2d(
            channels, 4, kernel_size=3, stride=1, padding=1
        )

        # # Init parameters.
        # prior_prob = cfg.MODEL.OneNet.PRIOR_PROB
        # self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self._reset_parameters()

    def init_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # initialize the bias for focal loss.
        nn.init.constant_(self.class_predictor.bias, self.bias_value)

    def forward(self, features: List[int]):
        class_logits_all: List[Tensor] = []
        bboxes_predictions_all: List[Tensor] = []

        for feature, feature_stride, scale in zip(
            features, self.features_stride, self.scales
        ):
            batch_size = feature.shape[0]
            # classes
            class_features = self.class_prediction_branch(feature)
            class_logits = self.class_predictor(class_features).view(
                batch_size, self.num_classes, -1
            )
            # bboxes
            regression_features = self.regression_prediction_branch(feature)
            center_points = self.get_center_points_on_grid(feature, feature_stride)[
                None
            ]
            # bboxes here are center points, so the coodinares are (cx,cy,h,w)
            bboxes_predictions = self.bboxes_predictor(regression_features)
            # rescale bboxes by a learnable parameter
            bboxes_predictions = scale(bboxes_predictions)
            # rescale bboxes based on the level stride
            bboxes_predictions = F.relu(bboxes_predictions) * feature_stride
            bboxes_predictions = self.to_xyxy_bboxes(
                center_points, bboxes_predictions
            ).view(batch_size, 4, -1)

            class_logits_all.append(class_logits)
            bboxes_predictions_all.append(bboxes_predictions)

        return torch.cat(class_logits_all, dim=-1), torch.cat(
            bboxes_predictions_all, dim=-1
        )

    def to_xyxy_bboxes(self, locations, pred_ltrb):
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
    def get_center_points_on_grid(self, features: Tensor, stride: int) -> Tensor:
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
        center_points = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        center_points = center_points.reshape(h, w, 2).permute(2, 0, 1)

        return center_points


if __name__ == "__main__":
    # layers = StackedConv2dLayers(32, 64)
    # print(layers)

    head = Head(256, channels=256)
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
