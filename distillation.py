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

from video_depth_anything.dpt import DepthAnythingV2
from video_depth_anything.video_depth import VideoDepthAnything
from dataset import create_depth_dataloaders
from utils.util import RunningAverageDict, RunningAverage

# Optional wandb import with handling
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

def SILogLoss(pred, target, mask=None, variance_focus=0.85):
    """
    Compute SILog loss between predicted and target depth maps.
    """
    if mask is None:
        mask = (target > 0).detach()

    mask[870:1016, 1570:1829] = 0

    pred = pred[mask]
    target = target[mask]

    log_diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
    mean_log_diff_squared = torch.mean(log_diff ** 2)
    mean_log_diff = torch.mean(log_diff)

    silog_loss = mean_log_diff_squared - variance_focus * (mean_log_diff ** 2)
    return silog_loss

def GradientMatchingLoss(pred, target, mask=None):
    """
    Compute gradient matching loss between predicted and target depth maps.
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"

    if mask is None:
        mask = (target > 0).detach()

    mask[870:1016, 1570:1829] = 0

    pred = torch.where(mask, pred, 0)
    target = torch.where(mask, target, 0)

    N = torch.sum(mask)
    log_d_diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)

    # print(f"Log Diff.: {torch.isnan(log_d_diff).any()}")

    v_grad = torch.abs(log_d_diff[...,:-2,:] - log_d_diff[..., 2:, :])
    h_grad = torch.abs(log_d_diff[..., :, :-2] - log_d_diff[..., :, 2:])

    # print(f"V. Grad: {torch.sum(v_grad)}; H. Grad: {torch.sum(h_grad)}")
    
    return (torch.sum(h_grad) + torch.sum(v_grad)) / N


def TemporalGradientMatchingLoss(pred, target, mask=None):
    assert pred.shape == target.shape, "pred and target must have the same shape"

    if mask is None:
        mask = (target > 0).detach()

    mask[:, 870:1016, 1570:1829] = 0
    mask.to(device)
    
    pred = torch.where(mask, pred, 0)
    target = torch.where(mask, target, 0)

    num_frames = pred.shape[0]
    time_mask = torch.abs(target[1:, :, :] - target[:-1, :, :]) > 0.05 # threshold decided by paper
    temporal_gm_loss = torch.nn.functional.l1_loss(
        (torch.abs(pred[1:, :, :] - pred[:-1, :, :])
         - torch.abs(target[1:, :, :] - target[:-1, :, :])) * time_mask) # ensure that we only sum values that make sense
    return temporal_gm_loss / (N - 1)


def depth_loss_criterion(preds, target, mask=None, variance_focus=0.85, alpha=2.0):
    """
    Combined depth estimation loss.
    """
    if alpha == 0.0:
        return SILogLoss(preds, target, mask, variance_focus)
    else:
        return SILogLoss(preds, target, mask, variance_focus) + alpha * GradientMatchingLoss(preds, target, mask)

def feature_distillation_loss(student_features, teacher_features, adaptation_layers=None, weights=None):
    """
    Compute MSE loss between student and teacher feature maps.
    Uses adaptation layers if provided, otherwise applies adaptive pooling
    if sizes don't match and optional weighting for different levels.

    Args:
        student_features: List of student feature maps
        teacher_features: List of teacher feature maps
        adaptation_layers: List of adaptation layers (or None) for each feature level
        weights: Optional list of weights for each feature level

    Returns:
        Total weighted feature distillation loss
    """
    if weights is None:
        # Default: higher weights for deeper features
        weights = [0.5, 1.0, 1.5, 2.0]

    total = sum(weights)
    weights = [w/total for w in weights]

    assert len(student_features) == len(teacher_features) == len(weights), \
        f"Mismatch in feature maps: student={len(student_features)}, teacher={len(teacher_features)}, weights={len(weights)}"

    total_loss = 0

    for i, (s_feat, t_feat, weight) in enumerate(zip(student_features, teacher_features, weights)):
        # Handle different spatial dimensions with adaptive pooling
        assert (s_feat.shape == t_feat.shape), "student and teacher feature maps have different shapes!"

        # Normalize features for better training stability
        s_feat = F.normalize(s_feat, p=2, dim=1)
        t_feat = F.normalize(t_feat, p=2, dim=1)

        # MSE loss between normalized features
        loss = F.mse_loss(s_feat, t_feat)
        total_loss += weight * loss

    return total_loss

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

def load_teacher_model(model_name: str, model_weights: Path, backbone_weights: Optional[Path]):
    """
    Load the teacher model (VideoDepthAnything).
    """
    model_config = MODEL_CONFIG[model_name]
    video_depth_anything = VideoDepthAnything(**model_config)

    # Load weights for the video depth model
    vda_weights = torch.load(model_weights, map_location=DEVICE, weights_only=True)
    if backbone_weights:
        # ensures that we share the same backbone for pretrained and finetuned models
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

@torch.autocast(device_type=DEVICE)
def distillation_train_step(
    student_model,
    teacher_model,
    train_dataloader,
    optimizer,
    scheduler,
    depth_criterion,
    distill_lambda=0.5,
    feature_weights=None,
    student_target_modules=None,
    teacher_target_modules=None,
    sub_batch_size=8
):
    """
    Training step with both depth supervision and feature distillation.
    Processes data in smaller sub-batches to reduce VRAM usage.
    
    Args:
        student_model: Student model to be trained
        teacher_model: Teacher model for knowledge distillation
        train_dataloader: DataLoader for training data
        optimizer: Optimizer for student model
        depth_criterion: Loss function for depth prediction
        distill_lambda: Weight for distillation loss vs depth loss
        feature_weights: Weights for different feature levels
        student_target_modules: List of module names in student to extract features from
        teacher_target_modules: List of module names in teacher to extract features from
        sub_batch_size: Size of sub-batches to process at once (to save VRAM)
    """
    student_model.train()
    teacher_model.eval()
    train_loss = RunningAverageDict()
    batch_train_loss = []
    scaler = torch.amp.GradScaler()
    
    for imgs, depth_maps, metadata in tqdm.tqdm(train_dataloader):
        if torch.isnan(imgs).any() or torch.isnan(depth_maps).any():
            print("NaN detected in inputs, skipping batch")
            continue
        
        # Move data to device
        imgs = imgs.to(DEVICE)  # [batch_size, frames, C, H, W]
        depth_maps = depth_maps.to(DEVICE).squeeze(1)  # [batch_size, frames, H, W]
        
        batch_size = imgs.shape[0]
        frames = imgs.shape[1]
        
        optimizer.zero_grad()
        
        # Initialize accumulated losses for the entire batch
        total_batch_loss = 0
        depth_batch_loss = 0
        distill_batch_loss = 0
        
        # Teacher forward pass (with no gradients)
        with torch.no_grad():
            teacher_output, teacher_features = teacher_model(
                imgs,
                get_temporal_maps=True, 
                run_efficiently=True
            )

        sub_batch_imgs = rearrange(imgs, "b (f m) c h w -> (b f) m c h w", m=sub_batch_size)
        sub_batch_depths = rearrange(depth_maps, "b (f m) h w -> (b f) m h w", m=sub_batch_size)
        sub_batch_teachers = rearrange(teacher_output, "b (f m) h w -> (b f) m h w", m=sub_batch_size)

        sub_batch_teacher_features = [rearrange(m, "(b f m) c h w -> (b f) m c h w", m=sub_batch_size, b=batch_size) for m in teacher_features]
        # reshape list to accomodate batches
        sub_batch_teacher_features = list(zip(*sub_batch_teacher_features))

        student_outputs = torch.zeros(*teacher_output.shape)
        # Process in sub-batches to save VRAM
        for idx, (sb_imgs, sb_depths, sb_teacher_features) in enumerate(zip(sub_batch_imgs, sub_batch_depths, sub_batch_teacher_features)):
            # Student forward pass
            student_output, student_features = student_model(
                sb_imgs,
                get_reassembled_maps=True
            )
            student_outputs[:, idx * sub_batch_size:(idx + 1) * sub_batch_size, :, :] = student_output
            
            # Calculate depth prediction loss
            # depth_loss = depth_criterion(student_output, sb_depths, alpha=0)
            depth_loss = torch.tensor(0.0)

            # Calculate feature distillation loss
            distill_loss = feature_distillation_loss(
                student_features,
                sb_teacher_features,
                weights=feature_weights
            )
            
            # Combined loss for this sub-batch (scaled by sub-batch size proportion)
            sub_batch_loss = (1 - distill_lambda) * depth_loss + distill_lambda * distill_loss
            # print(f"Depth Loss: {depth_loss} | Distill Loss: {distill_loss} | Subbatch Loss: {sub_batch_loss}")
            scaled_loss = sub_batch_loss
            
            # Accumulate scaled gradients
            scaler.scale(scaled_loss).backward()
            
            # Accumulate losses for logging
            total_batch_loss += sub_batch_loss.item()
            depth_batch_loss += depth_loss.item()
            distill_batch_loss += distill_loss.item()
            
            # Clear memory
            # del teacher_output, teacher_features, student_output, student_features
            # torch.cuda.empty_cache()
        
        # Update weights once per full batch
        temporal_loss = TemporalGradientMatchingLoss(student_outputs, teacher_output)
        scaler.scale(temporal_loss).backward()

        temporal_loss = temporal_loss.item()
        total_batch_loss += temporal_loss

        scaler.step(optimizer)
        scaler.update()
        
        # Record losses
        train_loss.update({
            "total_loss": total_batch_loss,
            "depth_loss": depth_batch_loss,
            "distill_loss": distill_batch_loss,
            "temporal_loss": temporal_loss
        })
        batch_train_loss.append(total_batch_loss)
    
    scheduler.step()
    return train_loss.get_value(), batch_train_loss

def distillation_eval_step(
    student_model,
    val_dataloader,
    depth_criterion,
    sport_name=None
):
    """
    Evaluation step for the student model.
    """
    from evaluate import evaluate as eval_depth_maps

    student_model.eval()
    metrics = RunningAverageDict()


    with torch.no_grad():
        for imgs, depth_maps, metadata in tqdm.tqdm(val_dataloader):
            imgs = imgs.to(DEVICE)
            depth_maps = depth_maps.to(DEVICE)
            depth_maps = rearrange(depth_maps, "b t h w -> (b t) h w")

            # Forward pass student model
            batch_size = imgs.shape[0]
            imgs_frames = rearrange(imgs, "b t c h w -> t b c h w")
            student_output_frames = []
            for frames in imgs_frames:
                student_output_frames.append(student_model(frames))
            student_output_frames = torch.cat(student_output_frames)
            # student_output = rearrange(student_output_frames, "(t b) h w -> b t h w", b=batch_size)

            # Calculate depth prediction loss
            depth_loss = depth_criterion(student_output_frames, depth_maps)
            metrics.update({'val_loss': depth_loss.item()})

            # Process each image in the batch
            for i in range(imgs.size(0)):
                batch_metrics = eval_depth_maps(
                    student_output_frames[i],
                    depth_maps[i],
                    sport_name=sport_name,
                    device=DEVICE,
                    mask_need=True
                )
                metrics.update(batch_metrics)

    return metrics.get_value()

def distillation_train(
    student_model,
    teacher_model,
    train_dataloader,
    val_dataloader,
    epochs,
    backbone_lr,
    dpt_head_lr,
    distill_lambda=0.5,
    feature_weights=None,
    use_wandb=False,
    sport_name=None,
    experiment_name=None,
    save_dir="saved_models"
):
    """
    Main distillation training function.

    Args:
        student_model: Student model to be trained
        teacher_model: Teacher model for knowledge distillation
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        epochs: Number of training epochs
        backbone_lr: Learning rate for backbone parameters
        dpt_head_lr: Learning rate for DPT head parameters
        distill_lambda: Weight for distillation loss vs depth loss
        feature_weights: Weights for different feature levels
        use_wandb: Whether to use Weights & Biases for logging
        sport_name: Optional sport name for evaluation metrics
        experiment_name: Name for this experiment
        save_dir: Directory to save model checkpoints
    """
    # Set up different parameter groups with different learning rates
    backbone_params = []
    head_params = []

    for name, param in student_model.named_parameters():
        if 'pretrained' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Set up optimizer with parameter groups
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': dpt_head_lr}
    ])
    min_lr = dpt_head_lr*1e-1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=min_lr  # This will be scaled for each param group
    )

    # Define loss function for depth prediction
    depth_criterion = depth_loss_criterion

    # Set up wandb if enabled
    if use_wandb and wandb is not None:
        wandb.init(project="depth_anything_v2_finetuning", name=experiment_name)
        wandb.config.update({
            "epochs": epochs,
            "backbone_lr": backbone_lr,
            "dpt_head_lr": dpt_head_lr,
            "distill_lambda": distill_lambda,
            "train_batch_size": train_dataloader.batch_size,
            "val_batch_size": val_dataloader.batch_size,
        })

    # Create directory to save models if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_val_loss = float('inf')
    best_abs_rel = float('inf')

    # Initial validation
    val_metrics = distillation_eval_step(student_model, val_dataloader, depth_criterion, sport_name)
    val_loss = val_metrics['val_loss']

    # Print metrics
    print(f"Epoch 0/{epochs}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Metrics:")
    for k, v in val_metrics.items():
        print(f"\t{k}: {v*1e3:.4f}")

    for epoch in range(epochs):
        # Training step with distillation
        train_loss, batch_train_loss = distillation_train_step(
            student_model,
            teacher_model,
            train_dataloader,
            optimizer,
            scheduler,
            depth_criterion,
            distill_lambda=distill_lambda,
            feature_weights=feature_weights
        )

        # Validation step
        val_metrics = distillation_eval_step(
            student_model,
            val_dataloader,
            depth_criterion,
            sport_name
        )
        val_loss = val_metrics['val_loss']

        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss['total_loss']:.4f}")
        print(f"Batch Train loss: {batch_train_loss}")
        print(f"Depth Loss: {train_loss['depth_loss']:.4f}")
        print(f"Distill Loss: {train_loss['distill_loss']:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics:")
        for k, v in val_metrics.items():
            print(f"\t{k}: {v*1e3:.4f}")

        # Log to wandb if enabled
        if use_wandb and wandb is not None:
            for loss in batch_train_loss:
                wandb.log({
                    "epoch": epoch + 1,
                    "batch_train_loss": loss
                })

            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss['total_loss'],
                "depth_loss": train_loss['depth_loss'],
                "distill_loss": train_loss['distill_loss'],
                **val_metrics
            }
            wandb.log(log_dict)

        # Save model if it's the best so far
        if val_loss < best_val_loss or val_metrics['abs_rel'] < best_abs_rel:
            best_val_loss = min(best_val_loss, val_loss)
            best_abs_rel = min(best_abs_rel, val_metrics['abs_rel'])
            model_path = os.path.join(save_dir, f"best_model_distilled_{experiment_name}.pth")
            torch.save(student_model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}_distilled_{experiment_name}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Close wandb run if it was used
    if use_wandb and wandb is not None:
        wandb.finish()

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
    Main function to set up and run distillation training.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Default experiment name if not provided
    if experiment_name is None:
        experiment_name = f"distill_{teacher_model_name}_to_{student_model_name}"

    # Load student model
    student_model = load_student_model(student_model_name, use_registers, student_model_weights)

    # Load teacher model
    teacher_model = load_teacher_model(teacher_model_name, teacher_model_weights, teacher_backbone_weights)

    # Create dataloaders
    train_dataloader, val_dataloader = create_depth_dataloaders(
        root_dir=dataset_root_path,
        sport_name=sport_name,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        seed=seed
    )

    # Default feature weights based on model size
    if student_model_name == teacher_model_name:
        # If same architecture, use balanced weights
        feature_weights = [1.0, 1.0, 1.0, 1.0]
    else:
        # If different architectures, focus on deeper features
        feature_weights = [0.5, 1.0, 1.5, 2.0]

    # Train with distillation
    distillation_train(
        student_model,
        teacher_model,
        train_dataloader,
        val_dataloader,
        epochs,
        backbone_lr,
        dpt_head_lr,
        distill_lambda=distill_lambda,
        feature_weights=feature_weights,
        use_wandb=use_wandb,
        sport_name=sport_name,
        experiment_name=experiment_name
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Distillation from Video to Image Depth Model')

    # Student model parameters
    parser.add_argument('--student-model', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'], default='vits',
                      help='Student model size to use')
    parser.add_argument('--student-weights', type=str, default=None,
                      help='Path to pre-trained student model weights (optional)')
    parser.add_argument('--use-registers', action='store_true',
                      help='Use DinoV2 backbone with registers for student')

    # Teacher model parameters
    parser.add_argument('--teacher-model', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'], default='vitl',
                      help='Teacher model size to use')
    parser.add_argument('--teacher-weights', type=str, required=True,
                      help='Path to teacher (video depth) model weights')
    parser.add_argument('--teacher-backbone-weights', type=str, required=False,
                        help='Path to teacher (video depth) backbone weights (adapting from finetuned depth models)')

    # Dataset parameters
    parser.add_argument('--dataset-path', type=Path, required=True,
                      help='Path to the dataset root directory')
    parser.add_argument('--sport-name', type=str, default=None,
                      help='Optional sport name filter for dataset')

    # Training parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--train-batch-size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=8,
                      help='Batch size for validation')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs')
    parser.add_argument('--backbone-lr', type=float, default=1e-6,
                      help='Learning rate for backbone parameters')
    parser.add_argument('--head-lr', type=float, default=1e-5,
                      help='Learning rate for DPT head parameters')
    parser.add_argument('--distill-lambda', type=float, default=0.5,
                      help='Weight for distillation loss vs depth loss (0-1)')

    # Logging and experiment parameters
    parser.add_argument('--use-wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Name for the experiment run')

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

