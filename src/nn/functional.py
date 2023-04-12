from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


def window_partition(x: Tensor, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
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
    # x = x.view(batch_size, pad_height // window_size, window_size, pad_weight // window_size, window_size, channels)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
    return windows, (pad_height, pad_weight), (height, width)


def window_unpartition(windows: Tensor, window_size: int, size: Tuple[int]) -> Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    height, width = size
    x = rearrange(
        windows, "(b wh ww) h w c -> b (h wh) (w ww) c", wh=window_size, ww=window_size
    ).contiguous()
    x = x[:, :height, :width, :].contiguous()
    return x


if __name__ == "__main__":
    import torch

    windows, heights = window_partition(torch.randn((1, 80, 80, 32)), window_size=14)
    x = window_unpartition(windows, window_size=14, size=(80, 80))
    print(windows.shape, heights, x.shape)
