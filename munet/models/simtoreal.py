"""Sim-to-Real Mapping Network using U-Net architecture with domain adaptation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BatchNorm -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SimToRealMapper(nn.Module):
    """
    U-Net based Sim-to-Real Mapping Network.
    
    This model maps synthetic (simulation) images to realistic images,
    preserving semantic content while adapting visual appearance.
    
    Args:
        n_channels: Number of input channels (default: 3 for RGB)
        n_classes: Number of output channels (default: 3 for RGB)
        bilinear: Use bilinear upsampling instead of transposed convolutions
        base_filters: Number of filters in the first layer (default: 64)
    """
    
    def __init__(self, n_channels=3, n_classes=3, bilinear=True, base_filters=64):
        super(SimToRealMapper, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, n_classes, height, width)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.tanh(logits)  # Output in range [-1, 1]

    def use_checkpointing(self):
        """
        Enable gradient checkpointing to save memory.
        Note: This needs to be called before training and will wrap forward passes.
        """
        # Gradient checkpointing is typically enabled by wrapping modules
        # in checkpoint during forward pass. This method is kept for API compatibility
        # but the actual implementation would require modifying the forward method.
        # For production use, consider using torch.utils.checkpoint.checkpoint()
        # directly in the forward method for memory-critical operations.
        import warnings
        warnings.warn(
            "use_checkpointing() is a placeholder. For actual gradient checkpointing, "
            "wrap module calls with torch.utils.checkpoint.checkpoint() in forward().",
            UserWarning
        )
