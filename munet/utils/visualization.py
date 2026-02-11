"""Visualization utilities for sim-to-real mapping."""

import torch
import torchvision.utils as vutils
from pathlib import Path


def denormalize(tensor: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    """
    Denormalize a tensor image.
    
    Args:
        tensor: Normalized image tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def save_comparison_grid(
    sim_images: torch.Tensor,
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    output_path: str,
    nrow: int = 4
):
    """
    Save a comparison grid showing sim, real, and generated images.
    
    Args:
        sim_images: Simulation images tensor (B, C, H, W)
        real_images: Real images tensor (B, C, H, W)
        generated_images: Generated images tensor (B, C, H, W)
        output_path: Path to save the grid image
        nrow: Number of images per row
    """
    # Denormalize images
    sim_denorm = denormalize(sim_images)
    real_denorm = denormalize(real_images)
    gen_denorm = denormalize(generated_images)
    
    # Clamp to [0, 1]
    sim_denorm = torch.clamp(sim_denorm, 0, 1)
    real_denorm = torch.clamp(real_denorm, 0, 1)
    gen_denorm = torch.clamp(gen_denorm, 0, 1)
    
    # Create comparison grid: [sim, generated, real] for each sample
    batch_size = sim_images.size(0)
    comparison = []
    for i in range(batch_size):
        comparison.append(sim_denorm[i])
        comparison.append(gen_denorm[i])
        comparison.append(real_denorm[i])
    
    comparison_tensor = torch.stack(comparison)
    
    # Save grid
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(comparison_tensor, output_path, nrow=nrow * 3, padding=2)
