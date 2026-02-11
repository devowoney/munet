"""Dataset implementation for image loading and preprocessing."""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Dataset for loading image pairs (input and target).

    Args:
        input_dir: Directory containing input images
        target_dir: Directory containing target images
        transform: Optional transform to apply to both images
        input_transform: Optional transform for input images only
        target_transform: Optional transform for target images only
    """

    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        transform: Optional[Callable] = None,
        input_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform
        self.input_transform = input_transform
        self.target_transform = target_transform

        # Get list of image files
        self.image_files = self._get_image_files()

    def _get_image_files(self) -> list:
        """Get list of matching image files in input and target directories."""
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        input_files = set()

        for f in self.input_dir.iterdir():
            if f.suffix.lower() in valid_extensions:
                input_files.add(f.stem)

        # Only include files that exist in both directories
        matched_files = []
        for stem in sorted(input_files):
            for ext in valid_extensions:
                target_path = self.target_dir / f"{stem}{ext}"
                if target_path.exists():
                    input_path = next(
                        (
                            self.input_dir / f"{stem}{e}"
                            for e in valid_extensions
                            if (self.input_dir / f"{stem}{e}").exists()
                        ),
                        None,
                    )
                    if input_path:
                        matched_files.append((input_path, target_path))
                    break

        return matched_files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_path, target_path = self.image_files[idx]

        # Load images
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # Apply shared transform
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        # Apply individual transforms
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target_image = self.target_transform(target_image)

        return input_image, target_image


def get_default_transforms(image_size: int = 256):
    """Get default transforms for training and validation."""
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform
