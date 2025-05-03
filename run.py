# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.
from typing import Optional
import pathlib
import argparse
import numpy as np
import os
import torch

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input-video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--max-res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max-len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target-fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save-npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save-exr', action='store_true', help='save depths as exr')

    # load backbone from different model
    parser.add_argument('--backbone-path', type=pathlib.Path, help='load [DepthAnything] backbone from different path?')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    vda_weights = torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu', weights_only=True)
    if args.backbone_path:
        # ensures that we share the same backbone for pretrained and finetuned models
        da_pretrained_weights = torch.load(args.backbone_path, map_location='cpu', weights_only=True)
        for k, v in da_pretrained_weights.items():
            if 'pretrained' in k:
                vda_weight = video_depth_anything.state_dict()[k]
                # print(f"replacing {k} in vda: before -- {torch.linalg.norm(vda_weight, dim=0, ord=2)}, after -- {torch.linalg.norm(v, dim=0, ord=2)}")
                vda_weights[k] = v
            elif 'head' in k:
                # ensure that we keep the names the same
                k = k.replace('depth_head', 'head')
                vda_weight = video_depth_anything.state_dict()[k]
                # print(f"replacing {k} in vda: before -- {torch.linalg.norm(vda_weight, dim=0, ord=2)}, after -- {torch.linalg.norm(v, dim=0, ord=2)}")
                vda_weights[k] = v
        print("replaced video depth anything backbone with depthanything!")
        video_depth_anything.load_state_dict(vda_weights, strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
    
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()

