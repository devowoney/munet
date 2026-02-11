# munet
sim-to-real mapping generative ML algorithm

## Overview

A UNet-based deep learning framework for image transformation tasks, using PyTorch with Hydra configuration management and TensorBoard monitoring.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
munet/
├── configs/                  # Hydra configuration files
│   ├── config.yaml          # Main configuration
│   ├── model/
│   │   └── unet.yaml        # Model configuration
│   └── training/
│       └── default.yaml     # Training configuration
├── src/
│   ├── models/
│   │   └── unet.py          # UNet architecture
│   ├── data/
│   │   └── dataset.py       # Dataset loaders
│   └── training/
│       └── trainer.py       # Training logic with TensorBoard
├── train.py                  # Main training script
└── requirements.txt          # Dependencies
```

## Usage

### Basic Training

```bash
python train.py
```

### Override Configuration

```bash
# Change learning rate and batch size
python train.py training.optimizer.lr=0.0001 data.batch_size=16

# Use different experiment name
python train.py experiment.name=my_experiment

# Train on CPU
python train.py hardware.device=cpu
```

### Multi-run Experiments

```bash
# Run multiple experiments with different learning rates
python train.py --multirun training.optimizer.lr=0.001,0.0001,0.00001
```

## Data Organization

Organize your data as follows:

```
data/
├── train/
│   ├── input/    # Training input images
│   └── target/   # Training target images
└── val/
    ├── input/    # Validation input images
    └── target/   # Validation target images
```

## Monitoring with TensorBoard

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir runs/
```

View metrics at: http://localhost:6006

### Logged Metrics

- Training/validation loss
- Learning rate schedule
- Sample input/output images

## Configuration

### Model (`configs/model/unet.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_channels` | 3 | Input image channels |
| `out_channels` | 3 | Output image channels |
| `base_features` | 64 | Base feature channels |
| `bilinear` | true | Use bilinear upsampling |

### Training (`configs/training/default.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Number of training epochs |
| `optimizer.lr` | 0.001 | Learning rate |
| `loss.name` | mse | Loss function (mse, l1, bce, dice) |

## License

MIT
