from enum import Enum
from typing import Any, Callable

import torch
import einops

class RunningAverage:
    def __init__(self):
        self.sum: Any = 0
        self.count: int = 0

    def append(self, value: Any):
        self.sum = value + self.sum
        self.count += 1

    def get_value(self) -> Any:
        return self.sum / self.count

class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict: dict[str, Any] = {}

    def update(self, new_dict: dict[str, Any]) -> None:
        for key, value in new_dict.items():
            if key not in self._dict:
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}

def compute_scale_and_shift_median(d_pred: torch.Tensor, d_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def __compute_scale_and_shift(d: torch.Tensor):
        # input: [B, M]
        
        # t(d) = median(d)
        # ensures that median is properly reduced
        t_d = d.quantile(0.5, dim=-1, keepdim=True)
        # s(d) = 1/M sum(|d - t_d|)
        s_d = (d - t_d).abs().mean(dim=-1, keepdim=True)
        
        # d^ = (d - t(d)) / s(d)
        return (d - t_d) / s_d

    return __compute_scale_and_shift(d_pred), __compute_scale_and_shift(d_gt)

def compute_scale_and_shift_ols(d_pred: torch.Tensor, d_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # input: [B, M] tensors

    # we can simplify (D^T D)^(-1) D^T D* to the following expression:
    # s = cov(D, D*) / var(D) = E[(D - mean(D))(D* - mean(D*))] / E[(D - mean(D))^2]
    # t = mean(D*) - s * mean(D)

    mu_pred = d_pred.mean(dim=-1, keepdim=True)
    mu_gt = d_gt.mean(dim=-1, keepdim=True)

    d_pred_centered = d_pred - mu_pred
    d_gt_centered = d_gt - mu_gt

    # covariance calculation
    cov = (d_pred_centered * d_gt_centered).sum()
    var = (d_pred_centered ** 2).sum()

    s = cov / var
    t = mu_gt - s * mu_pred

    d_pred_correct = s * d_pred + t
    return d_pred_correct, d_gt

def compute_scale_and_shift_ols_faithful(d_pred: torch.Tensor, d_gt: torch.Tensor):
    # input: [B, M] tensors

    eps = 1e-8
    
    ones = torch.ones_like(d_pred)

    # Build D = [d_pred, 1] -> shape (B, M, 2)
    D = einops.rearrange([d_pred, ones], 'n b m -> b m n')

    # Compute DᵀD and Dᵀy using einops
    D_t = einops.rearrange(D, 'b m n -> b n m')
    D_t_D = torch.matmul(D_t, D)                 # (B, 2, 2)
    D_t_y = torch.matmul(D_t, d_gt.unsqueeze(-1))# (B, 2, 1)

    # Add tiny epsilon to diagonal for stability
    D_t_D = D_t_D + eps * torch.eye(2, device=D.device).unsqueeze(0)

    # Solve for h = (DᵀD)⁻¹ Dᵀy
    h = torch.linalg.solve(D_t_D, D_t_y)
    s = h[:, 0, 0]
    t = h[:, 1, 0]

    # Aligned output
    d_aligned = s.unsqueeze(1) * d_pred + t.unsqueeze(1)

    return d_aligned, d_pred

class ScaleAndShiftMethod(Enum):
    OLS_FAST = 1
    MEDIAN = 2
    OLS_FAITHFUL = 3

def compute_errors(d_pred: torch.Tensor, d_gt: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    # print("number of 0s", (d_gt == 0).sum())
    # print(f"nans: gt -- {torch.isnan(d_gt).sum()} | pred -- {torch.isnan(d_pred).sum()}")
    # print(f"infs: gt -- {torch.isinf(d_gt).sum()} | pred -- {torch.isinf(d_pred).sum()}")

    return {
        # "absrel": ((d_pred - d_gt).abs() / d_gt).sum() / mask.sum(),
        "rmse": (d_pred - d_gt).pow(2).sum() / mask.sum()
    }
    
def evaluate(d_pred: torch.Tensor, d_gt: torch.Tensor, mask: torch.Tensor, method: ScaleAndShiftMethod) -> dict[str, torch.Tensor]:
    assert d_pred.shape == d_gt.shape and d_gt.shape == mask.shape

    mask = torch.bitwise_or(mask, d_gt > 0)
    eps = 1e-6

    d_pred_valid = torch.where(mask, d_pred, eps)
    d_pred_valid = einops.rearrange(d_pred_valid, 'b c h -> b (c h)')
    d_gt_valid = torch.where(mask, d_gt, eps)
    d_gt_valid = einops.rearrange(d_gt_valid, 'b c h -> b (c h)')

    sas_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    
    match method:
        case ScaleAndShiftMethod.OLS_FAST:
            sas_fn = compute_scale_and_shift_ols
        case ScaleAndShiftMethod.MEDIAN:
            sas_fn = compute_scale_and_shift_median
        case ScaleAndShiftMethod.OLS_FAITHFUL:
            sas_fn = compute_scale_and_shift_ols_faithful

    d_pred_adjusted, d_gt_adjusted = sas_fn(d_pred_valid, d_gt_valid)


    return compute_errors(d_pred_adjusted, d_gt_adjusted, mask)
