#!/bin/bash
#!/usr/bin/env bash
# Usage: ./download.sh {small|base|large|huge}

model="$1"
[ -z "$model" ] && echo "Usage: $0 {small|base|large}" && exit 1

case "$model" in
    small)
	url="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth";
	params="45M" ;;
    base)
	url="https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth";
	params="150M";;
    large)
	url="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth";
	params="500M" ;;
  *) echo "Invalid option"; exit 1 ;;
esac

echo "Model: $model"
echo "Parameters: $params"
sz=$(curl -sIL "$u" | awk '/content-length/ {print $2}' | tr -d '\r | tail -n 1')
[ -n "$sz" ] && echo "Size: $((sz/1024/1024)) MB" || echo "Size: unknown"

mkdir -p checkpoints && cd checkpoints
wget "$url"
