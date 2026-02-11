"""Evaluation script for sim-to-real mapping with Hydra configuration."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from munet.models import SimToRealMapper
from munet.data import SimToRealDataset
from munet.utils import calculate_psnr, calculate_ssim, save_comparison_grid

logger = logging.getLogger(__name__)


def load_model(cfg: DictConfig, device: torch.device) -> SimToRealMapper:
    """Load trained model from checkpoint."""
    model = SimToRealMapper(
        n_channels=cfg.model.n_channels,
        n_classes=cfg.model.n_classes,
        bilinear=cfg.model.bilinear,
        base_filters=cfg.model.base_filters
    )
    
    checkpoint_path = cfg.evaluation.checkpoint_path
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        logger.info(f"Model trained for {checkpoint['epoch']} epochs")
    
    return model


def setup_dataloader(cfg: DictConfig):
    """Setup test dataloader."""
    test_dataset = SimToRealDataset(
        root=cfg.data.test_dir,
        img_size=tuple(cfg.data.img_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    
    return test_loader


@torch.no_grad()
def evaluate(
    model: SimToRealMapper,
    test_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device
):
    """Evaluate the model on test set."""
    model.eval()
    
    # Create output directory
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics storage
    metrics = {
        'psnr': [],
        'ssim': []
    }
    
    logger.info("Starting evaluation...")
    
    for batch_idx, (sim_imgs, real_imgs) in enumerate(tqdm(test_loader, desc="Evaluating")):
        sim_imgs = sim_imgs.to(device)
        real_imgs = real_imgs.to(device)
        
        # Generate images
        generated = model(sim_imgs)
        
        # Calculate metrics
        if cfg.evaluation.calculate_metrics:
            if 'psnr' in cfg.evaluation.metrics:
                psnr = calculate_psnr(generated, real_imgs)
                metrics['psnr'].append(psnr)
            
            if 'ssim' in cfg.evaluation.metrics:
                ssim = calculate_ssim(generated, real_imgs)
                metrics['ssim'].append(ssim)
        
        # Save images
        if cfg.evaluation.save_images:
            output_path = output_dir / f"comparison_{batch_idx:04d}.png"
            save_comparison_grid(
                sim_imgs,
                real_imgs,
                generated,
                str(output_path),
                nrow=cfg.data.batch_size
            )
    
    # Calculate and log average metrics
    if cfg.evaluation.calculate_metrics:
        logger.info("\n" + "="*50)
        logger.info("Evaluation Results:")
        logger.info("="*50)
        
        if 'psnr' in cfg.evaluation.metrics and metrics['psnr']:
            avg_psnr = sum(metrics['psnr']) / len(metrics['psnr'])
            logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
        
        if 'ssim' in cfg.evaluation.metrics and metrics['ssim']:
            avg_ssim = sum(metrics['ssim']) / len(metrics['ssim'])
            logger.info(f"Average SSIM: {avg_ssim:.4f}")
        
        logger.info("="*50)
        
        # Save metrics to file
        metrics_file = output_dir / "metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("Evaluation Metrics\n")
            f.write("="*50 + "\n")
            if 'psnr' in cfg.evaluation.metrics and metrics['psnr']:
                f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            if 'ssim' in cfg.evaluation.metrics and metrics['ssim']:
                f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        logger.info(f"Metrics saved to {metrics_file}")
    
    if cfg.evaluation.save_images:
        logger.info(f"Comparison images saved to {output_dir}")


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(cfg, device)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Please train a model first using train.py")
        return
    
    # Setup dataloader
    try:
        test_loader = setup_dataloader(cfg)
    except Exception as e:
        logger.warning(f"Could not load test dataset: {e}")
        logger.warning("Creating dummy data directory for demonstration...")
        # Create dummy directory if data doesn't exist
        test_dir = cfg.data.test_dir
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        Path(test_dir, 'sim').mkdir(exist_ok=True)
        Path(test_dir, 'real').mkdir(exist_ok=True)
        logger.info("Please add your test data to the created directories and rerun evaluation.")
        return
    
    # Evaluate
    evaluate(model, test_loader, cfg, device)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
