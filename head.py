from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation

from functools import partial

# Conv2dLayerNormGELU = partial(
#     Conv2dNormActivation, norm_layer=nn.LayerNorm, activation_layer=nn.GELU
# )

class Conv2dLayerNormGELU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels),
        self.act = nn.GELU()


class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale

class StackedConv2dLayers(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2dLayerNormGELU(in_channels, out_channels, kernel_size=3),
            *[Conv2dLayerNormGELU(out_channels, out_channels, kernel_size=3) for _ in range(depth - 1)]
        )

class Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,  depth: int = 2, num_classes: int = 80):
        super().__init__()

  
        num_levels = 4

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_levels)])
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features_stride = [8, 16, 32, 64, 128]

        # [NOTE] we may need RELU in the last layer to force  0 - 1 
        self.class_prediction_branch = StackedConv2dLayers(in_channels, out_channels, depth)
        self.regression_prediction_branch = StackedConv2dLayers(in_channels, out_channels, depth)

        # was cls_score
        self.class_score_prediction = nn.Conv2d(
            out_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        # was ltrb_pred
        self.points_prediction = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)

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
        nn.init.constant_(self.cls_score.bias, self.bias_value)

    def forward(self, features_list):
        class_logits = list()
        pred_bboxes = list()
        locationall = list()
        fpn_levels = list()

        for l, stride_feat in enumerate(features_list):
            cls_feat = stride_feat
            reg_feat = stride_feat

            for conv_layer in self.cls_conv_module:
                cls_feat = conv_layer(cls_feat)

            for conv_layer in self.reg_conv_module:
                reg_feat = conv_layer(reg_feat)

            locations = self.locations(stride_feat, self.features_stride[l])[None]

            stride_class_logits = self.cls_score(cls_feat)
            reg_ltrb = self.ltrb_pred(reg_feat)

            scale_reg_ltrb = self.scales[l](reg_ltrb)
            stride_pred_ltrb = F.relu(scale_reg_ltrb) * self.features_stride[l]
            stride_pred_bboxes = self.apply_ltrb(locations, stride_pred_ltrb)
            bs, c, h, w = stride_class_logits.shape
            bs, four, h, w = stride_pred_bboxes.shape
            class_logits.append(stride_class_logits.view(bs, c, -1))
            pred_bboxes.append(stride_pred_bboxes.view(bs, four, -1))

        class_logits = torch.cat(class_logits, dim=-1)
        pred_bboxes = torch.cat(pred_bboxes, dim=-1)

        return class_logits, pred_bboxes

    def apply_ltrb(self, locations, pred_ltrb):
        """
        here we are rescaling the grid by pred_ltrb
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
    def locations(self, features, stride):
        """
        This code essentially computes the (x, y) coordinates of the center points of evenly spaced cells in a grid, given the height and width of the grid and the stride between the cells.

        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        device = features.device
        # if stride is 8
        # [0, 8, 16, 24 ...]
        shifts_x = torch.arange(
            0, w * stride, step=stride, dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        locations = locations.reshape(h, w, 2).permute(2, 0, 1)

        return locations

if __name__ == "__main__":
    layers = StackedConv2dLayers(32, 64)
    print(layers)

    head = Head()
    head(torch.randn((1, 256, 40, 40)),
        torch.randn((1, 256, 20, 20)),
        torch.randn((1, 256, 10, 10)),
        torch.randn((1, 256, 5, 5)),)