"""Tests for UNet model."""

import torch

from munet.models.unet import UNet


def test_unet_output_shape():
    """UNet output should match spatial dims and output channels."""
    model = UNet(in_channels=3, out_channels=3, base_features=32, bilinear=True)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 3, 64, 64)


def test_unet_single_channel():
    """UNet should handle single-channel inputs and outputs."""
    model = UNet(in_channels=1, out_channels=1, base_features=16, bilinear=True)
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    assert y.shape == (1, 1, 32, 32)


def test_unet_transposed_conv():
    """UNet should work with transposed convolutions instead of bilinear upsampling."""
    model = UNet(in_channels=3, out_channels=3, base_features=32, bilinear=False)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 3, 64, 64)
