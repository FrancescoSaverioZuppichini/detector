# copied from CLIP
# https://github.com/openai/CLIP/blob/main/clip/model.py

import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torchvision.ops import StochasticDepth

from ..types import Backbone
from .common import QuickGELU
from .functional import window_partition, window_unpartition


class LayerNorm(nn.LayerNorm):
    """Subclass torch's nn.LayerNorm to handle fp16.
    [EDIT] this shouldn't be needed if we use autocast
    see https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        drop_rate: float = 0.0,
        window_size: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.drop_path = StochasticDepth(drop_rate, mode="batch")
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        shortcut = x
        if self.window_size > 0:
            # [NOTE] we need to rearrange everything here
            x, pad_hw, hw = window_partition(x, self.window_size)
        x = self.attention(self.ln_1(x))
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, hw)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]

        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(width, heads, attn_mask, drop_rate=drop_rates[i])
                for i in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# modied CLIP ViT to follow our Backbone Interface
class ViT(Backbone):
    def __init__(
        self,
        input_resolution: List[int],
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        # [NOTE] it was self.input_resolution = input_resolution
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution[0] // patch_size) * (input_resolution[1] // patch_size) + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # [NOTE] changed from original implementation, we skip the cls token
        return [x[:, 1:, :]]

    # Copied from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L174  and adapter
    def interpolate_pos_encoding(self, w, h):
        N = self.positional_embedding.shape[0] - 1
        if self.input_resolution[0] == w and self.input_resolution[1] == h:
            return self.positional_embedding
        class_positional_embedding = self.positional_embedding[0,...]
        patch_positional_embedding = self.positional_embedding[1:,...]
        dim = self.positional_embedding.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_positional_embedding = nn.functional.interpolate(
            patch_positional_embedding.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_positional_embedding.shape[-2]
            and int(h0) == patch_positional_embedding.shape[-1]
        )
        patch_positional_embedding = patch_positional_embedding.permute(
            0, 2, 3, 1
        ).view(1, -1, dim)
        return torch.cat(
            (class_positional_embedding[None, None, ...], patch_positional_embedding), dim=1
        )
