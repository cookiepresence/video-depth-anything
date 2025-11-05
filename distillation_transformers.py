# goal:
# take video model -- done
# infer videos under that
# save each individual inference with image numbers
# load each image, train on image model
# profit?

import random
from typing import Callable, Optional
from pathlib import Path
import os

import einops
import numpy as np
import torch
from tqdm import tqdm
import wandb
import transformers
import h5py

from datasets import point_odessey, image_hdf5
from video_depth_anything.dpt import DepthAnythingV2
from video_depth_anything.video_depth import VideoDepthAnything

MODEL_CONFIG = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vits_r': {'encoder': 'vits_r', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42

def set_seed(seed):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = cudnn_benchmark
    # torch.backends.cudnn.deterministic = cudnn_deterministic
import torch
import torch.nn.functional as F

def SILogLoss(
        gt: torch.Tensor,
        pred: torch.Tensor,
        rho: Callable[[torch.Tensor], torch.Tensor] = lambda x: x ** 2
):
    return rho(pred - gt).mean()


def GradientMatchingLoss(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor = None):
    """
    Computes L1 gradient difference between pred and gt.
    |∂x(pred - gt)| + |∂y(pred - gt)|
    """
    residual = pred - gt

    # gradient in x direction (difference between adjacent columns)
    grad_x = residual[:, :, 1:] - residual[:, :, :-1]
    # gradient in y direction (difference between adjacent rows)
    grad_y = residual[:, 1:, :] - residual[:, :-1, :]

    loss = grad_x.abs().mean() + grad_y.abs().mean()
    return loss

def loss_criterion(gt: torch.Tensor, pred: torch.Tensor, masks: torch.Tensor, alpha: float = 2.0):
    def scale_and_shift(d: torch.Tensor):
        t = d.median()
        s = (d - t).abs().mean()
        return (d - t) / s

    if torch.sum(masks) > 0:
        gt = torch.where(masks, gt, 0)
        pred = torch.where(masks, pred, 0)

        gt_scaled = scale_and_shift(gt)
        pred_scaled = scale_and_shift(pred)

        sil = torch.nan_to_num(SILogLoss(gt_scaled, pred_scaled))
        grad = torch.nan_to_num(GradientMatchingLoss(gt_scaled, pred_scaled))

        return sil + alpha * grad
    else:
        return None

def load_vda(model_name: str, model_weights: Path, backbone_weights: Optional[Path]):
    """
    Load the teacher model (VideoDepthAnything).
    """
    model_config = MODEL_CONFIG[model_name]
    video_depth_anything = VideoDepthAnything(**model_config)

    # Load weights for the video depth model
    vda_weights: dict[str, torch.Tensor] = torch.load(model_weights, map_location=DEVICE, weights_only=True)
    if backbone_weights:
        # ensures that we share the same backbone for pretrained and finetuned models
        da_pretrained_weights: dict[str, torch.Tensor] = torch.load(backbone_weights, map_location='cpu', weights_only=True)
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
    _ = video_depth_anything.load_state_dict(vda_weights, strict=True)

    video_depth_anything = video_depth_anything.to(DEVICE)
    return video_depth_anything


def load_depthanything(model_family: str, model_weights: Optional[str]):
    # model_config = MODEL_CONFIG[model_name]
    # model_path = f'checkpoints/depth_anything_v2_{model_name}.pth'
    # depth_anything = DepthAnythingV2(**model_config)
    # if model_weights is not None:
    #     _ = depth_anything.load_state_dict(torch.load(model_weights, weights_only=True))
    # else:
    #     _ = depth_anything.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

    # depth_anything = depth_anything.to(DEVICE)
    # return depth_anything
    
    model_weights = model_family if model_weights is None else model_weights
    model = transformers.AutoModelForDepthEstimation.from_pretrained(model_weights)
    return model


def run_vda_step(vda_model, train_dataloader, save_path: Path):
    vda_model = vda_model.to(DEVICE).eval()
    counter = 0

    h5_file = h5py.File(save_path, "w")
    datasets_created = False
    img_dset: Optional[h5py.Dataset] = None
    disparity_dset: Optional[h5py.Dataset] = None
    mask_dset: Optional[h5py.Dataset] = None

    with torch.no_grad():
        for (imgs, disparity_gt, masks) in tqdm(train_dataloader):
            imgs: torch.Tensor = imgs.to(DEVICE)
            disparity_gt: torch.Tensor = disparity_gt.to(DEVICE)
            masks: torch.Tensor = masks.to(DEVICE)

            # ensure we mask gts where they are 0
            # dim: [B, S, C, H, W]
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                disparity_preds: torch.Tensor = vda_model(imgs, over_imgs=False)
            video_to_imgs: Callable[[torch.Tensor], torch.Tensor] = lambda tensor: einops.rearrange(tensor, 'b s h w -> (b s) h w')

            # convert all videos to imgs
            # [b, s, h, w] -> [n, h, w]
            imgs = einops.rearrange(imgs, 'b s c h w -> (b s) c h w')
            disparity_preds = video_to_imgs(disparity_preds)
            disparity_gt = video_to_imgs(disparity_gt)
            masks = video_to_imgs(masks)

            # save all images
            imgs_np = imgs.cpu().numpy()
            preds_np = disparity_preds.cpu().numpy()
            gt_np = disparity_gt.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Create datasets on first iteration
            if not datasets_created:
                total_images = len(train_dataloader.dataset) * train_dataloader.dataset.sequence_len
                img_dset = h5_file.create_dataset(
                    "images",
                    shape=(total_images,) + imgs_np.shape[1:],
                    dtype=imgs_np.dtype,
                    chunks=(4,) + imgs_np.shape[1:]
                    # compression="lzf"
                )
                disparity_dset = h5_file.create_dataset(
                    "disparity_preds",
                    shape=(total_images,) + preds_np.shape[1:],
                    dtype=preds_np.dtype,
                    chunks=(4,) + preds_np.shape[1:]
                    # compression="lzf"
                )
                mask_dset = h5_file.create_dataset(
                    "masks",
                    shape=(total_images,) + masks_np.shape[1:],
                    dtype=masks_np.dtype,
                    chunks=(4,) + masks_np.shape[1:]
                    # compression="lzf"
                )
                datasets_created = True

            # Write to datasets
            batch_size = imgs_np.shape[0]
            img_dset[counter:counter + batch_size] = imgs_np
            disparity_dset[counter:counter + batch_size] = preds_np
            mask_dset[counter:counter + batch_size] = masks_np
            counter += batch_size

    h5_file.close()

def run_train_step(
        da_model: transformers.DepthAnythingForDepthEstimation,
        train_dataloader,
        model_save_path: Path
):
    da_model = da_model.to(DEVICE)
    # optimizer = torch.optim.AdamW(da_model.head.parameters(), lr=5e-5)
    # optimizer = torch.optim.AdamW(da_model.neck.parameters(), lr=5e-5)

    optimizer = torch.optim.AdamW([{'params' : da_model.head.parameters()},
                                    {'params' : da_model.neck.parameters()},],
                                    lr=5e-5)

    # actual training loop
    for epoch, (imgs, vid_disparities, masks) in enumerate(tqdm(train_dataloader)):
        imgs = imgs.to(DEVICE)
        vid_disparities = vid_disparities.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        img_disparities = da_model(imgs).predicted_depth
        # TODO: write out loss function
        # TODO: log loss function using wandb
        # TODO: remove images that have all 0s using masks
        loss = loss_criterion(vid_disparities, img_disparities, masks)
        if loss is None:
            continue
        # print(loss)
        wandb.log({"loss": loss})
        loss.backward()

        total_norm = torch.sqrt(
            sum(
                (p.grad.detach() ** 2).sum()
                for submodule in [da_model.head, da_model.neck]
                for p in submodule.parameters()
                if p.grad is not None
            )
        )
        wandb.log({"grad_norm": total_norm})
        optimizer.step()

        # save model after every 1000 epochs
        if epoch % 1000 == 0:
            torch.save(da_model.state_dict(), model_save_path)
    torch.save(da_model.state_dict(), model_save_path)

set_seed(SEED)

# vid_train_dataset = point_odessey.PointOdessey(
#     data_root=Path('/scratch/mde/pointodessey'),
#     splits='train'
# )

# video_train_dataloader = torch.utils.data.DataLoader(
#     vid_train_dataset,
#     batch_size=4,
#     shuffle=False,
#     num_workers=2,
#     pin_memory=True,
# )

# vda = load_vda(model_name='vits', model_weights=Path('checkpoints/video_depth_anything_vits.pth'), backbone_weights=None)
depth_anything = load_depthanything(model_family='depth-anything/Depth-Anything-V2-Small-hf', model_weights=None)

save_path = Path('/scratch/mde/test.hdf5')
# run_vda_step(vda_model=vda, train_dataloader=video_train_dataloader, save_path=save_path)

img_train_dataset = image_hdf5.ImageHdf5Dataset(file_path=save_path)
img_train_dataloader = torch.utils.data.DataLoader(
    img_train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=16,
    pin_memory=True
)

run = wandb.init(
    project="depthanything-distillation",
    config={
        "head_lr": 5e-5,
        "batch_size": 16
    })
run_train_step(da_model=depth_anything, train_dataloader=img_train_dataloader, model_save_path=Path('/scratch/mde/transformers-run-01.pt'))
