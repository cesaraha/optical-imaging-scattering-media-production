# src/losses.py
import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy loss.
    Computes pos_weight per batch from the target itself,
    fixing the global pre-computation bug from the thesis.

    pos_weight = number of negative pixels / number of positive pixels
    This upweights the rare foreground (white pattern) pixels.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # compute pos_weight per batch
        n_pos = target.sum() + self.eps
        n_neg = (1.0 - target).sum() + self.eps
        pos_weight = n_neg / n_pos

        return nn.functional.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=torch.tensor(pos_weight, device=pred.device),
        )


class CombinedLoss(nn.Module):
    """
    Weighted BCE + SSIM loss.
    alpha controls the balance: loss = alpha * BCE + (1 - alpha) * (1 - SSIM)
    Default alpha=0.8 prioritises BCE while SSIM regularises structure.
    """

    def __init__(self, alpha: float = 0.8, eps: float = 1e-8):
        super().__init__()
        self.alpha   = alpha
        self.bce     = WeightedBCELoss(eps)
        self.eps     = eps

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # BCE expects logits, SSIM expects sigmoid output
        bce_loss  = self.bce(pred, target)
        pred_sig  = torch.sigmoid(pred)
        ssim_loss = 1.0 - ssim_metric(pred_sig, target, eps=self.eps)
        return self.alpha * bce_loss + (1.0 - self.alpha) * ssim_loss


def ssim_metric(pred: torch.Tensor, target: torch.Tensor,
                window_size: int = 11, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable SSIM for use in both loss and metric.
    pred and target: (B, 1, H, W), values in [0, 1].
    Returns mean SSIM across the batch.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # gaussian kernel
    kernel = _gaussian_kernel(window_size, sigma=1.5,
                               device=pred.device,
                               dtype=pred.dtype)

    mu_x    = _conv(pred,   kernel)
    mu_y    = _conv(target, kernel)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    sigma_x  = _conv(pred   * pred,   kernel) - mu_x_sq
    sigma_y  = _conv(target * target, kernel) - mu_y_sq
    sigma_xy = _conv(pred   * target, kernel) - mu_xy

    num  = (2 * mu_xy   + C1) * (2 * sigma_xy + C2)
    den  = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    ssim = (num / den.clamp(min=eps))

    return ssim.mean()


def _gaussian_kernel(size: int, sigma: float,
                     device, dtype) -> torch.Tensor:
    coords  = torch.arange(size, device=device, dtype=dtype) - size // 2
    g       = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g       = g / g.sum()
    kernel  = g[:, None] * g[None, :]
    return kernel.view(1, 1, size, size)


def _conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return nn.functional.conv2d(x, kernel, padding=padding, groups=x.shape[1])