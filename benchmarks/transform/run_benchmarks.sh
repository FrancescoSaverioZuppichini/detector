#!/bin/bash
# thanks daddyGTP
NAMES=("pytorch-vanilla" "pytorch-gpu")
data="/home/zuppif/Documents/neatly/detector/datasets/train/images"
# List of base sizes to iterate over
BATCH_SIZES=(8 16 32 64)
NUM_WORKERS=(4 8 16)

for size in "${BATCH_SIZES[@]}"
do
    for workers in "${NUM_WORKERS[@]}"
    do
            echo "Running benchmarks with batch size $size with workers $workers"
            # python benchmark.py pytorch-vanilla $data  --batch-size "$size" --image-size 640 640 --num-workers $workers --device cpu
            python benchmark.py pytorch-vanilla $data  --batch-size "$size" --image-size 640 640 --num-workers $workers --device cuda
            # python benchmark.py pytorch-gpu $data  --batch-size "$size" --image-size 640 640 --num-workers $workers --device cpu
            python benchmark.py pytorch-gpu $data  --batch-size "$size" --image-size 640 640 --num-workers $workers --device cuda
            python benchmark.py pytorch-gpu $data  --batch-size "$size" --image-size 640 640 --num-workers $workers --device cuda --compile
    done
done