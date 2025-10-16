#!/bin/bash
#!/usr/bin/env bash
# Usage: ./download.sh {small|base|large|huge}

model="$1"
[ -z "$model" ] && echo "Usage: $0 {small|base|large|huge}" && exit 1

case "$model" in
    small)
	url="https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth";
	params="21M" ;;
    large)
	url="https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth";
	params="381M" ;;
  *) echo "Invalid option"; exit 1 ;;
esac

echo "Model: $model"
echo "Parameters: $params"
sz=$(curl -sI "$u" | awk '/content-length/ {print $2}' | tr -d '\r')
[ -n "$sz" ] && echo "Size: $((sz/1024/1024)) MB" || echo "Size: unknown"

mkdir -p checkpoints && cd checkpoints
wget "$url"
