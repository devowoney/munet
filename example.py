"""Example usage script for MuNet sim-to-real mapping."""

import torch
from munet.models import SimToRealMapper
from munet.data import SimToRealDataset
from munet.utils import calculate_psnr, calculate_ssim


def example_model_usage():
    """Demonstrate basic model usage."""
    print("=" * 60)
    print("Example: Model Initialization and Forward Pass")
    print("=" * 60)
    
    # Initialize model
    model = SimToRealMapper(
        n_channels=3,
        n_classes=3,
        bilinear=True,
        base_filters=64
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy input
    batch_size = 4
    height, width = 256, 256
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print()


def example_metrics_usage():
    """Demonstrate metrics calculation."""
    print("=" * 60)
    print("Example: Metrics Calculation")
    print("=" * 60)
    
    # Create dummy images
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    
    # Calculate metrics
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print()


def example_configuration_override():
    """Show how to use Hydra configuration overrides."""
    print("=" * 60)
    print("Example: Configuration Overrides")
    print("=" * 60)
    
    print("Training with default config:")
    print("  python train.py")
    print()
    
    print("Training with custom parameters:")
    print("  python train.py \\")
    print("    training.epochs=200 \\")
    print("    training.learning_rate=0.0001 \\")
    print("    data.batch_size=16 \\")
    print("    model.base_filters=128")
    print()
    
    print("Evaluation with custom checkpoint:")
    print("  python eval.py \\")
    print("    evaluation.checkpoint_path=checkpoints/model_epoch_100.pth \\")
    print("    evaluation.output_dir=results/epoch_100")
    print()


def example_model_variants():
    """Show different model configurations."""
    print("=" * 60)
    print("Example: Model Variants")
    print("=" * 60)
    
    configs = [
        {"base_filters": 32, "bilinear": True, "desc": "Lightweight model"},
        {"base_filters": 64, "bilinear": True, "desc": "Default model"},
        {"base_filters": 128, "bilinear": False, "desc": "Large model with transposed conv"},
    ]
    
    for config in configs:
        model = SimToRealMapper(
            n_channels=3,
            n_classes=3,
            base_filters=config["base_filters"],
            bilinear=config["bilinear"]
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{config['desc']}: {param_count:,} parameters")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MuNet Example Usage")
    print("=" * 60 + "\n")
    
    example_model_usage()
    example_metrics_usage()
    example_model_variants()
    example_configuration_override()
    
    print("=" * 60)
    print("For full training and evaluation, see:")
    print("  - train.py: Training script")
    print("  - eval.py: Evaluation script")
    print("  - README.md: Complete documentation")
    print("=" * 60)


if __name__ == "__main__":
    main()
