from enum import Enum
from typing import Callable

from .yolo import get_image_and_labels as yolo_get_image_and_labels


class FormatType(Enum):
    YOLO = "yolo"


def get_parser(format_type: FormatType) -> Callable:
    # [NOTE] maybe fix types in return?
    if format_type == FormatType.YOLO:
        return yolo_get_image_and_labels
    else:
        raise NotImplementedError(f"format_type={format_type} unknown.")
