import sys
sys.path.append('/home/zuppif/Documents/neatly/detector/')
from src.data.datasets.yolo import YOLODataset
from pathlib import Path
from src.data.transform import SequentialAugmentation, RandomHFlip
import torch
from src.data.data import ObjectDetectionData

# merging all annotations in one
dec = msgspec.json.Decoder(COCOFile)
root = Path("/Users/giuliocesare/Documents/ODinW-RF100-challenge/rf100-test-coco")
out_filename = Path(
    "/Users/giuliocesare/Documents/ODinW-RF100-challenge/annotations/odinw_rf100_annotations_testsplit.json"
)
filenames = root.glob("**/*.json")

coco_annotations: Dict[str, List[COCOFile]] = {}
for filename in tqdm(filenames):
    with filename.open("r") as f:
        dataset_name = filename.parent.stem
        print(filename)
        coco_file = dec.decode(f.read())
        coco_annotations[dataset_name] = [coco_file]

with out_filename.open("wb") as f:
    f.write(msgspec.json.encode(coco_annotations))