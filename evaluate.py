import torch

from utils.util import compute_scale_and_shift_isl as compute_scale_and_shift, compute_errors, compute_metrics, RunningAverageDict

def evaluate(preds, gts, mask, device):
    mask.to(device)

    # i=0
    with torch.no_grad():
        # for preds, gts in zip(pred_loader, gt_loader):
        preds, gts = preds.to(device), gts.to(device)

        scale, shift = compute_scale_and_shift(preds, gts, mask)
        scaled_predictions = scale.view(-1, 1, 1) * preds + shift.view(-1, 1, 1)

        return compute_metrics(preds, gts, mask)
