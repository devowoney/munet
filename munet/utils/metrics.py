"""Metrics for evaluating sim-to-real mapping quality."""

import torch
import torch.nn.functional as F


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: First image tensor of shape (B, C, H, W) or (C, H, W)
        img2: Second image tensor of shape (B, C, H, W) or (C, H, W)
        max_val: Maximum possible pixel value (default: 1.0)
        
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image tensor of shape (B, C, H, W)
        img2: Second image tensor of shape (B, C, H, W)
        window_size: Size of the Gaussian window (default: 11)
        size_average: Whether to average over all dimensions
        
    Returns:
        SSIM value between -1 and 1 (higher is better)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, 1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, 1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()
