# src/metrics.py
import torch
from src.losses import ssim_metric


def pearson_r(pred: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation coefficient, computed per image then averaged.
    pred and target: (B, 1, H, W), values in [0, 1].
    """
    B    = pred.shape[0]
    p    = pred.view(B, -1)
    t    = target.view(B, -1)
    pm   = p - p.mean(dim=1, keepdim=True)
    tm   = t - t.mean(dim=1, keepdim=True)
    num  = (pm * tm).sum(dim=1)
    den  = (pm.pow(2).sum(dim=1) * tm.pow(2).sum(dim=1)).sqrt().clamp(min=eps)
    return (num / den).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor,
         eps: float = 1e-8) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio.
    pred and target: (B, 1, H, W), values in [0, 1].
    Returns mean PSNR in dB across the batch.
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return torch.mean(10 * torch.log10(1.0 / mse.clamp(min=eps)))


def compute_all_metrics(pred_logits: torch.Tensor,
                        target: torch.Tensor) -> dict:
    """
    Convenience function called at validation time.
    pred_logits: raw model output (before sigmoid)
    target:      binary ground truth (B, 1, H, W)
    Returns dict of scalar metric values.
    """
    pred = torch.sigmoid(pred_logits)
    return {
        "ssim":     ssim_metric(pred, target).item(),
        "pearson":  pearson_r(pred, target).item(),
        "psnr":     psnr(pred, target).item(),
    }