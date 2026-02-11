"""Training loop with TensorBoard logging."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for UNet sim-to-real mapping.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        lr: Learning rate.
        epochs: Number of training epochs.
        checkpoint_dir: Directory to save model checkpoints.
        log_dir: Directory for TensorBoard logs.
        device: Device to train on (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        epochs: int = 100,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs",
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_epoch(self, epoch: int) -> float:
        """Run a single training epoch. Returns the average training loss."""
        self.model.train()
        total_loss = 0.0
        for batch_idx, (sim, real) in enumerate(self.train_loader):
            sim, real = sim.to(self.device), real.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(sim)
            loss = self.criterion(output, real)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation. Returns the average validation loss."""
        self.model.eval()
        total_loss = 0.0
        for sim, real in self.val_loader:
            sim, real = sim.to(self.device), real.to(self.device)
            output = self.model(sim)
            loss = self.criterion(output, real)
            total_loss += loss.item()
        return total_loss / max(len(self.val_loader), 1)

    def train(self) -> None:
        """Run the full training loop."""
        best_val_loss = float("inf")
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            logger.info(
                "Epoch %d/%d - train_loss: %.6f - val_loss: %.6f",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    self.checkpoint_dir / "best_model.pt",
                )

        self.writer.close()
        logger.info("Training complete. Best val loss: %.6f", best_val_loss)

    def save_checkpoint(self, path: str) -> None:
        """Save a model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
