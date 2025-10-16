from typing import Optional
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tqdm

from datasets import point_odessey
import evaluate
from utils.util import RunningAverageDict, RunningAverage
from video_depth_anything.video_depth import VideoDepthAnything


MODEL_CONFIG = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vits_r': {'encoder': 'vits_r', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_imgs(src, pred, gt, mask, name):
    """
    Plots an RGB image, predicted and ground truth depth maps (as disparity), 
    applying a mask to exclude invalid pixels, and saves the plot.

    Args:
        src (Tensor): RGB image tensor of shape [C, H, W]
        pred (Tensor): Predicted disparity tensor of shape [H', W']
        gt (Tensor): Ground truth disparity tensor of shape [H', W']
        mask (Tensor): Binary mask tensor of shape [H', W'] (1 = valid, 0 = ignore)
        name (str): Output filename (saved to 'artifacts/name.png')
    """
    def normalize_tensor(tensor, valid_mask):
        """Normalizes a 2D tensor to [0, 1] using only valid masked values."""
        arr = tensor.detach().cpu().numpy()
        msk = valid_mask.detach().cpu().numpy().astype(bool)

        valid_vals = arr[msk]
        if valid_vals.size > 0:
            min_val = np.min(valid_vals)
            max_val = np.max(valid_vals)
            if max_val > min_val:
                norm_arr = (arr - min_val) / (max_val - min_val)
            else:
                norm_arr = np.zeros_like(arr)
        else:
            norm_arr = np.zeros_like(arr)
        return norm_arr
        # return np.log(norm_arr, where=msk)

    # Convert RGB image to HWC format
    src_np = src.detach().cpu().numpy()
    if src_np.shape[0] == 3:
        src_np = np.transpose(src_np, (1, 2, 0))
    src_np = np.clip(src_np, 0, 1)

    # Convert disparity to depth using inverse
    # gt_depth = 300 / (gt + 1e-6)
    # pred_depth = torch.max(pred) - pred + 1e6
    # print(torch.mean(torch.abs(pred_depth - gt) / gt))

    # Normalize using mask
    # pred_norm = normalize_tensor(pred, mask)
    # gt_norm = normalize_tensor(gt, mask)
    # diff_norm = normalize_tensor(torch.abs(pred - gt), mask)

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(src_np)
    axs[0].set_title("RGB Image")
    axs[1].imshow(pred.cpu().numpy(), cmap='gray')
    axs[1].set_title("Predicted Depth")
    axs[2].imshow(gt.cpu().numpy(), cmap='gray')
    axs[2].set_title("Ground Truth Depth")
    axs[3].imshow(torch.abs(pred - gt).cpu().numpy(), cmap='gray')
    axs[3].set_title("Difference between pred and ground truth")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'artifacts/{name:05}.png')
    print(f"saving file in artifacts/{name:05}.png")
    plt.close(fig)


def load_vda(model_name: str, model_weights: Path, backbone_weights: Optional[Path]):
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

def evaluate_vda(model, dataset):
    model.eval()
    metrics = RunningAverageDict()
    
    with torch.no_grad():
        for (imgs, disparity_gt, masks) in tqdm.tqdm(dataset):
            print(imgs.shape, disparity_gt.shape, masks.shape)
            imgs = imgs.to(DEVICE)
            disparity_gt = disparity_gt.to(DEVICE)
            masks = masks.to(DEVICE)

            # ensure we mask gts where they are 0
            # dim: [B, S, C, H, W]
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                disparity_preds = model(imgs, over_imgs=False)

            video_to_imgs = lambda tensor: einops.rearrange(tensor, 'b s h w -> (b s) h w')

            # convert all videos to imgs
            # [b, s, h, w] -> [n, h, w]
            imgs = einops.rearrange(imgs, 'b s c h w -> (b s) c h w')
            disparity_preds = video_to_imgs(disparity_preds)
            disparity_gt = video_to_imgs(disparity_gt)
            masks = video_to_imgs(masks)

            # currently: calculate evals over disparity
            # ideally: disparity -> (pseudo)-depth, evals over depth range
            batch_size = imgs.shape[0]

            # Evaluation is performed at the original resolution by interpolating the prediction
            disparity_preds = torchvision.transforms.functional.resize(
                disparity_preds,
                disparity_gt.shape[-2:],
                torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True
            )

            print("predictions done!")
            
            # pass to evaluate func
            for i in range(disparity_gt.shape[0]):
                batch_metrics = evaluate.evaluate(
                    disparity_preds[i].clone(),
                    disparity_gt[i].clone(),
                    device=DEVICE,
                    mask=masks[i].clone()
                )
                if 'empty' in batch_metrics:
                    continue

                if batch_metrics['rmse'] > 10: # TODO: remove this
                    plot_imgs(imgs[i], disparity_preds[i], disparity_gt[i], masks[i], 0 * 32 + i)
                    print(f"overall: {metrics.get_value()}")

                metrics.update(batch_metrics)
                print(f"current: {batch_metrics}")
            # break
            del disparity_preds
            del disparity_gt
            del imgs
            del masks
    return metrics.get_value()

vda = load_vda(model_name='vits', model_weights=Path('checkpoints/video_depth_anything_vits.pth'), backbone_weights=None)

# TODO: make sure this is correctly cropped to ensure that
# the model works correctly
eval_dataset = point_odessey.PointOdessey(
    data_root='/scratch/mde/pointodessey',
    splits='val',
)
# ensure that training-related augmentations are disabled
eval_dataset.eval()
print(len(eval_dataset))
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    drop_last=False,
    prefetch_factor=1
)

metrics = evaluate_vda(vda, eval_dataloader)
print("evaluation done!")
print(metrics)
# NOTE: this should work for things to move ahead
# checkpoint vals:
# AbsRel: 0.086
