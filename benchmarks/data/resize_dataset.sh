# !/bin/bash
src_dir=$1
out_dir=$2

echo "src_dir=$src_dir"
mkdir -p "$out_dir"

for f in $(ls $src_dir); do
    filename=$(basename "$f")
    ffmpeg -i "$src_dir/$f" -vf "scale=640:640" "$out_dir/$filename"
done