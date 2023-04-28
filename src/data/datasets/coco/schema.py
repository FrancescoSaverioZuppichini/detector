from typing import List, Dict
from msgspec import Struct
import msgspec


class COCOInfo(Struct):
    year: str
    version: str
    description: str
    contributor: str
    url: str
    date_created: str


class COCOLicense(Struct):
    id: int
    url: str
    name: str


class COCOCategory(Struct):
    id: int
    name: str
    supercategory: str


class COCOImage(Struct):
    id: int
    license: int
    file_name: str
    height: int
    width: int
    date_captured: str


class COCOAnnotation(Struct):
    id: int
    image_id: int
    category_id: int
    bbox: List[float] = list()
    area: float = 0
    segmentation: List[List[float]] = list()
    iscrowd: int = 0


class COCOFile(Struct):
    info: COCOInfo
    licenses: List[COCOLicense]
    categories: List[COCOCategory]
    images: List[COCOImage]
    annotations: List[COCOAnnotation]


CocoEvalAnnotations = Dict[str, List[COCOFile]]


class CocoResult(Struct):
    image_id: int
    category_id: int
    bbox: List[float] = list()
    score: float = 0.0


CocoZeroShotSubmission = Dict[str, List[List[CocoResult]]]

# CocoFewZeroShotResult
