"""
Module defining Pseudo-3D building blocks for the network. This is implementation of the Pseudo-3D convolutional layer
 from the paper "Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks"
(https://arxiv.org/abs/1711.10305).
"""
from abc import ABC

import torch
import torch.nn as nn


class P3DBlock(nn.Module, ABC):
    def __init__(self, in_channels: int, inside_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int | tuple[int, int, int] = 1, dilation: int | tuple[int, int, int] = 1,
                 bias: bool = False, use_batchnorm: bool = True) -> None:
        super().__init__()

        use_shortcut_reduction = stride != 1 or dilation != 1 or in_channels != out_channels

        self.shortcut_reduction = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                            dilation=dilation, bias=bias) if use_shortcut_reduction else nn.Identity()

        self.bottleneck_downsample = nn.Conv3d(in_channels, inside_channels, kernel_size=1, stride=stride, padding=0,
                                               dilation=dilation, bias=bias)
        self.bn_downsample = nn.BatchNorm3d(inside_channels) if use_batchnorm else nn.Identity()

        self.spatial_conv = nn.Conv3d(inside_channels, inside_channels, kernel_size=(1, kernel_size, kernel_size),
                                      padding="same", bias=bias)
        self.bn_spatial = nn.BatchNorm3d(inside_channels) if use_batchnorm else nn.Identity()

        self.temporal_conv = nn.Conv3d(inside_channels, inside_channels, kernel_size=(kernel_size, 1, 1),
                                       padding="same", bias=bias)
        self.bn_temporal = nn.BatchNorm3d(inside_channels) if use_batchnorm else nn.Identity()

        self.bottleneck_upsample = nn.Conv3d(inside_channels, out_channels, kernel_size=1, padding="same", bias=bias)
        self.bn_upsample = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # Implementation should be in subclasses


class P3DBlockTypeA(P3DBlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.bottleneck_downsample(x)
        x = self.bn_downsample(x)
        x = self.relu(x)

        x = self.spatial_conv(x)
        x = self.bn_spatial(x)
        x = self.relu(x)

        x = self.temporal_conv(x)
        x = self.bn_temporal(x)
        x = self.relu(x)

        x = self.bottleneck_upsample(x)
        x = self.bn_upsample(x)

        x += self.shortcut_reduction(identity)
        x = self.relu(x)

        return x

class P3DBlockTypeB(P3DBlock):
    """
    Pseudo-3D ResNet-like block of type B.

    This type of block performs temporal and spatial convolutions in parallel and then adds the results with a
    residual connection, as described in the paper. There is no direct connection between the spatial and temporal
    convolutional layers.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass type B.

        :param x: Input tensor.
        :return: Output tensor.
        """
        identity = x

        x = self.bottleneck_downsample(x)
        x = self.bn_downsample(x)
        x = self.relu(x)

        x_spatial = self.spatial_conv(x)
        x_spatial = self.bn_spatial(x_spatial)
        x_spatial = self.relu(x_spatial)

        x_temporal = self.temporal_conv(x)
        x_temporal = self.bn_temporal(x_temporal)
        x_temporal = self.relu(x_temporal)

        x = x_spatial + x_temporal

        x = self.bottleneck_upsample(x)
        x = self.bn_upsample(x)

        x += self.shortcut_reduction(identity)
        x = self.relu(x)

        return x


class P3DBlockTypeC(P3DBlock):
    """
    Pseudo-3D ResNet-like block of type C.

    This type of block is a combination of type A and type B. It performs temporal and spatial convolutions in parallel
    and adds the results, but temporal convolution is fed with the output of the spatial convolution. This type of
    block also uses a residual connection.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass type C.

        :param x: Input tensor.
        :return: Output tensor.
        """
        identity = x

        x = self.bottleneck_downsample(x)
        x = self.bn_downsample(x)
        x = self.relu(x)

        x_spatial = self.spatial_conv(x)
        x_spatial = self.bn_spatial(x_spatial)
        x_spatial = self.relu(x_spatial)

        x_temporal = self.temporal_conv(x_spatial)
        x_temporal = self.bn_temporal(x_temporal)
        x_temporal = self.relu(x_temporal)

        x = x_spatial + x_temporal

        x = self.bottleneck_upsample(x)
        x = self.bn_upsample(x)

        x += self.shortcut_reduction(identity)
        x = self.relu(x)

        return x

class AttentionBlock3D(nn.Module):
    def __init__(self, num_filters: int, reduction_ratio: int = 4) -> None:
        super().__init__()

        self.shared_mlp_channel_attention = nn.Sequential(
            nn.Conv3d(num_filters, num_filters // reduction_ratio, kernel_size=1, stride=1, padding="same", bias=False),
            nn.ReLU(),
            nn.Conv3d(num_filters // reduction_ratio, num_filters, kernel_size=1, stride=1, padding="same", bias=False)
        )

        self.conv_spatial_attention = nn.Conv3d(num_filters * 2, 1, kernel_size=7, stride=1, padding="same", bias=False)

        self.channel_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.channel_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=2, keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv_spatial_attention(x)
        x = self.sigmoid(x)
        return x

    def forward_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        x_avg = self.channel_avg_pool(x)
        x_max = self.channel_max_pool(x)
        x = self.shared_mlp_channel_attention(x_avg) + self.shared_mlp_channel_attention(x_max)
        x = self.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.forward_channel_attention(x) * x
        x = self.forward_spatial_attention(x) * x
        x = x + identity
        return x