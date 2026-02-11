# MuNet: Sim-to-Real Mapping Generative ML Algorithm

MuNet is a PyTorch-based deep learning framework for mapping synthetic (simulation) images to realistic images using a U-Net architecture. The project leverages Hydra for flexible configuration management, making it easy to experiment with different training and evaluation settings.

## Features

- **U-Net Based Architecture**: Powerful encoder-decoder network for image-to-image translation
- **PyTorch Implementation**: Built on PyTorch for efficient GPU acceleration
- **Hydra Configuration**: Flexible YAML-based configuration system
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Evaluation Metrics**: PSNR and SSIM metrics for quality assessment
- **TensorBoard Integration**: Real-time training visualization
- **Paired and Unpaired Datasets**: Support for both paired and unpaired sim-to-real data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/devowoney/munet.git
cd munet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Project Structure

```
munet/
├── munet/                  # Main package
│   ├── models/            # Model architectures
│   │   └── simtoreal.py  # U-Net based sim-to-real mapper
│   ├── data/              # Dataset implementations
│   │   └── dataset.py    # Paired and unpaired datasets
│   └── utils/             # Utility functions
│       ├── metrics.py    # PSNR and SSIM metrics
│       └── visualization.py  # Image visualization utilities
├── configs/               # Hydra configuration files
│   ├── train/            # Training configurations
│   │   └── config.yaml
│   └── eval/             # Evaluation configurations
│       └── config.yaml
├── train.py              # Training script
├── eval.py               # Evaluation script
└── requirements.txt      # Python dependencies
```

## Data Preparation

Organize your data in the following structure:

```
data/
├── train/
│   ├── sim/              # Simulated training images
│   │   ├── img1.png
│   │   └── ...
│   └── real/             # Real training images
│       ├── img1.png
│       └── ...
├── val/
│   ├── sim/              # Simulated validation images
│   └── real/             # Real validation images
└── test/
    ├── sim/              # Simulated test images
    └── real/             # Real test images
```

For paired datasets, ensure that corresponding images in `sim/` and `real/` have the same filename.

## Usage

### Training

Train the model using default configuration:
```bash
python train.py
```

Override configuration parameters:
```bash
python train.py training.epochs=200 training.learning_rate=0.0001 data.batch_size=16
```

Specify custom configuration file:
```bash
python train.py --config-name=custom_config
```

### Evaluation

Evaluate a trained model:
```bash
python eval.py evaluation.checkpoint_path=checkpoints/best_model.pth
```

Evaluate with custom settings:
```bash
python eval.py evaluation.checkpoint_path=checkpoints/best_model.pth data.test_dir=data/custom_test
```

## Configuration

### Training Configuration (`configs/train/config.yaml`)

Key parameters:
- `model`: Model architecture settings (channels, filters, etc.)
- `data`: Dataset paths and data loading settings
- `training`: Training hyperparameters (learning rate, epochs, etc.)
- `checkpoint`: Checkpoint saving settings
- `logging`: TensorBoard and logging configuration

### Evaluation Configuration (`configs/eval/config.yaml`)

Key parameters:
- `model`: Model architecture (should match training config)
- `data`: Test dataset path and settings
- `evaluation`: Checkpoint path, output directory, metrics to calculate

## Model Architecture

MuNet uses a U-Net architecture with:
- **Encoder**: 5 downsampling blocks with skip connections
- **Decoder**: 4 upsampling blocks
- **Features**: Batch normalization, ReLU activation, and optional bilinear upsampling
- **Output**: Tanh activation for normalized outputs in range [-1, 1]

The model is designed for sim-to-real domain adaptation, preserving semantic content while adapting visual appearance.

## Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality in dB
- **SSIM (Structural Similarity Index)**: Measures perceptual similarity between images

## Results

Training results are saved to:
- `checkpoints/`: Model checkpoints
- `logs/`: TensorBoard logs and comparison images
- `results/`: Evaluation results and generated images

View training progress with TensorBoard:
```bash
tensorboard --logdir=logs
```

## Advanced Usage

### Custom Loss Functions

Extend the training script to add custom loss functions:
```python
from munet.models import SimToRealMapper
# Add your custom loss implementation
```

### Fine-tuning

Resume training from a checkpoint:
```bash
python train.py checkpoint.resume=checkpoints/model_epoch_50.pth
```

### Multi-GPU Training

The code is ready for multi-GPU training with DataParallel:
```python
model = nn.DataParallel(model)
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full list of dependencies

## License

MIT License

## Citation

If you use this code in your research, please cite:
```bibtex
@software{munet2026,
  title={MuNet: Sim-to-Real Mapping Generative ML Algorithm},
  author={MuNet Team},
  year={2026},
  url={https://github.com/devowoney/munet}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- U-Net architecture inspired by Ronneberger et al. (2015)
- Built with PyTorch and Hydra frameworks
