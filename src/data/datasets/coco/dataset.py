from functools import reduce
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.data.data import ObjectDetectionData
