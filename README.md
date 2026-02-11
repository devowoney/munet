# MuNet

Sim-to-real mapping generative ML algorithm based on UNet, built with PyTorch, Hydra, and TensorBoard.

## Project Structure

```
munet/
├── configs/                # Hydra configuration
│   ├── config.yaml         # Default config (composes model + data + training)
│   ├── model/
│   │   └── unet.yaml       # UNet architecture hyperparameters
│   ├── data/
│   │   └── default.yaml    # Dataset paths, splits, and loader settings
│   └── training/
│       └── default.yaml    # Learning rate, epochs, device, logging
├── munet/                  # Python package
│   ├── models/
│   │   └── unet.py         # UNet model implementation
│   ├── data/
│   │   └── dataset.py      # xarray-based SimRealDataset
│   ├── training/
│   │   └── trainer.py      # Training loop with TensorBoard logging
│   └── utils/              # Shared utilities
├── train.py                # Hydra entry point
└── pyproject.toml          # Project metadata and dependencies
```

## Setup

```bash
pip install -e ".[dev]"
```

## Usage


### Train the model

```bash
python train.py
```

Override any hyperparameter via the command line:

```bash
python train.py training.lr=0.0005 training.epochs=50 model.base_features=32
```

### Monitor with TensorBoard

```bash
tensorboard --logdir runs
```
