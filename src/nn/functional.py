from typing import Tuple

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def window_partition(
    x: Tensor, window_size: int
) -> Tuple[Tensor, Tuple[int], Tuple[int]]:
    """
    # [CHECK] double check this
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (Tensor of shape `(batch_size, height, width, channels)`): Input tokens.
        window_size (int): The target window size.
    Returns:
        windows (Tensor of shape `(batch_size * num_windows, num_windows, num_windows, channels)`): Windows after partition.
        (pad_height, pad_weight): Padded height and width **before** partition.
        (height, width): Original height and width.

    """
    _, height, width, _ = x.shape

    pad_right = (window_size - height % window_size) % window_size
    pad_bottom = (window_size - width % window_size) % window_size
    should_pad = pad_right > 0 or pad_bottom > 0
    if should_pad:
        x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
    pad_height, pad_weight = height + pad_right, width + pad_bottom
    # where ws is windows size
    windows = rearrange(
        x, "b (h wh) (w ww) c -> (b wh ww) h w c", wh=window_size, ww=window_size
    ).contiguous()
    return windows, (pad_height, pad_weight), (height, width)


def window_unpartition(
    windows: Tensor, window_size: int, before_padding_size: Tuple[int]
) -> Tensor:
    """
    Window unpartition into original sequences and removing padding. Inverse of `window_partition`.
    Args:
        x (Tensor of shape `(batch_size * num_windows, num_windows, num_windows, channels)`): Input partitioned tokens.
        window_size (int): Window size.
        before_padding_size (Tuple): Height and width before padding..
    Returns:
        x (Tensor of shape `(batch_size, height, width, channels)`): Unpartitioned sequences.
    """
    height, width = before_padding_size
    x = rearrange(
        windows, "(b wh ww) h w c -> b (h wh) (w ww) c", wh=window_size, ww=window_size
    ).contiguous()
    x = x[:, :height, :width, :].contiguous()
    return x


if __name__ == "__main__":
    import torch

    windows, padded_size, original_size = window_partition(
        torch.randn((1, 80, 80, 32)), window_size=14
    )
    x = window_unpartition(windows, window_size=14, before_padding_size=(80, 80))
    print(windows.shape, padded_size, original_size, x.shape)
