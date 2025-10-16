from typing import List, Literal, Dict, Any, Optional
from pathlib import Path
import argparse
import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from video_depth_anything.dpt import DepthAnythingV
from video_depth_anything.video_depth import VideoDepthAnything
from dataset import create_depth_dataloaders
from utils.util import RunningAverageDict, RunningAverage

try: 
    import wandb
except ImportError:
    wandb = None

MODEL_CONFIG = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vits_r': {'encoder': 'vits_r', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def SILogLoss()

def GradientMatchingLoss()

def TemporalGradientMatchingLoss()

def depth_loss_criterion()

def load_student_model(model_name: str, use_registers: bool = False, model_weights: str = None):
    """
    Load the student model (DepthAnythingV2).
    """
    model_name_r = model_name + '_r' if use_registers else model_name
    model_config = MODEL_CONFIG[model_name_r]
    model_path = f'checkpoints/depth_anything_v2_{model_name}.pth'

    depth_anything = DepthAnythingV2(**model_config)

    if model_weights is not None:
        depth_anything.load_state_dict(torch.load(model_weights, weights_only=True))
    else:
        depth_anything.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

    if use_registers:
        depth_anything.pretrained.load_state_dict(
            torch.load(f'checkpoints/dinov2-with-registers-{model_name}.pt', map_location=DEVICE, weights_only=True)
        )

    depth_anything = depth_anything.to(DEVICE)
    return depth_anything

def load_teacher_model(model_name: str, model_weights: str, backbone_weights: str = None):
    """
    Load the teacher model (VideoDepthAnything).
    """
    model_config = MODEL_CONFIG[model_name]
    video_depth_anything = VideoDepthAnything(**model_config)

    # Load weights for the video depth model
    vda_weights = torch.load(model_weights, map_location=DEVICE, weights_only=True)
    if backbone_weights:
        # ensures that we share the same backbone for pretrained and finetuned models
        #! this might not be necessary now since we are not using finetuned backbone anymore.
        da_pretrained_weights = torch.load(backbone_weights, map_location='cpu', weights_only=True)
        for k, v in da_pretrained_weights.items():
            if 'pretrained' in k:
                vda_weight = video_depth_anything.state_dict()[k]
                vda_weights[k] = v
            elif 'head' in k:
                # ensure that we keep the names the same
                k = k.replace('depth_head', 'head')
                vda_weight = video_depth_anything.state_dict()[k]
                vda_weights[k] = v
        print("replaced video depth anything backbone with depthanything!")
    video_depth_anything.load_state_dict(vda_weights, strict=True)

    # Set to evaluation mode and don't compute gradients for teacher
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    for param in video_depth_anything.parameters():
        param.requires_grad = False

    return video_depth_anything

def distillation_train(
    student_model, 
    teacher_model, 
    train_dataloader, 
    val_dataloader,
    epochs, 
    backbone_lr, 
    dpt
)

def main(
    student_model_name: Literal['vits', 'vitb', 'vitl', 'vitg'],
    teacher_model_name: Literal['vits', 'vitb', 'vitl', 'vitg'],
    teacher_model_weights: Path,
    teacher_backbone_weights: Path,
    dataset_root_path: Path,
    student_model_weights: str = None,
    sport_name: str = None,
    seed: int = 42,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    epochs: int = 30,
    backbone_lr: float = 1e-5,
    dpt_head_lr: float = 1e-4,
    distill_lambda: float = 0.5,
    use_wandb: bool = False,
    experiment_name: str = None,
    use_registers: bool = False
):
    """
    Main function to set up and run the training process for distilling a student model from a teacher model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if experiment_name is None:
        experiment_name = f"distill_{student_model_name}_from_{teacher_model_name}_{os.path.basename(dataset_root_path)}"

    # our student model would be a DepthAnythingV2 model which has been initialized with pretrained weights. 
    student_model = load_student_model(student_model_name, use_registers, student_model_weights)

    # our teacher model would be the video depth anything model.
    teacher_model = load_teacher_model(teacher_model_name, teacher_model_weights, teacher_backbone_weights)

    # ! monish create dataloader. 
    train_dataloader, val_dataloader = distillation_dataloaders()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a student model to distill knowledge from a teacher model for video depth anything.')
    parser = setup_training_args(parser)
    args = parser.parse_args()

    main(
        student_model_name=args.student_model,
        teacher_model_name=args.teacher_model,
        teacher_model_weights=args.teacher_weights,
        teacher_backbone_weights=args.teacher_backbone_weights,
        dataset_root_path=args.dataset_path,
        student_model_weights=args.student_weights,
        sport_name=args.sport_name,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        backbone_lr=args.backbone_lr,
        dpt_head_lr=args.head_lr,
        distill_lambda=args.distill_lambda,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name,
        use_registers=args.use_registers
    )


