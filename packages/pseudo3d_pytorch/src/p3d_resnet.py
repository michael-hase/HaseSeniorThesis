"""
Implementation of Pseudo-3D ResNet-like network for 3D image/video data classification.

This implementation is inspired by the Pseudo-3D ResNet from the paper
"Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks" (https://arxiv.org/abs/1711.10305).
"""

from itertools import cycle

import torch
import torch.nn as nn
import torchsummary

from pseudo3d_pytorch.src.blocks import P3DBlockTypeA, P3DBlockTypeB, P3DBlockTypeC


class P3DResnet(nn.Module):
    """
    Class implementing the Pseudo-3D ResNet-like network.

    The network consists of an initial convolutional block (7x7x7) and a pooling layer (3x3x3), followed by four stages
    of Pseudo-3D ResNet-like blocks (3x3x3) (see: blocks.py). Network size is determined by the number of blocks in
    each stage. The output of the final stage is passed through a global average pooling layer and a fully
    connected layer to produce the final output of the network.
    """

    def __init__(self, input_channels: int, num_classes: int, block_type: str = "sequential",
                 num_blocks_per_stage: tuple[int, ...] = (3, 4, 6, 3), dropout_value: float | None = 0.3,
                 use_batchnorm: bool = True, base_channels: int = 64) -> None:
        """
        Initialize the Pseudo-3D ResNet-like network.

        :param input_channels: Number of input channels.
        :param num_classes: Number of classes in the dataset. This is the number of neurons in the output layer.
        :param block_type: "A", "B", "C" or "sequential". If "sequential", the network will be built using all three
        block types in the order A, B, C, A, (...). Else, the network will be built using only with specified block.
        :param num_blocks_per_stage: Number of blocks in each of the four stages of the network. The initial stage
        is the initial convolutional layer (7x7x7) and the pooling layer. After that, there are four stages,
        each consisting of a (a, b, c, d) blocks. Default is (3, 4, 6, 3), which corresponds to the ResNet-50
        architecture. For ResNet-101, use (3, 4, 23, 3). For ResNet-152, use (3, 8, 36, 3).
        :param dropout_value: Dropout value to use after the last fully connected layer. Default is 0.3.
        :param use_batchnorm: Whether to use batch normalization in the network. Default is True.
        :param base_channels: Base number of channels in the network. As default, the base number of channels is 64 for
        ResNets. It is possible to scale network up or down "horizontally" by changing this number.
        """
        super().__init__()

        assert block_type in ("A", "B", "C", "sequential"), "block_type must be one of 'A', 'B', 'C' or 'sequential'"
        for num_blocks in num_blocks_per_stage:
            assert num_blocks > 0, "num_blocks_per_stage must be a tuple of positive integers"
        assert input_channels > 0, "input_channels must be a positive integer"
        assert num_classes > 0, "num_classes must be a positive integer"

        self.block_type = block_type
        self.block = self._get_block_generator()
        self.input_channels = input_channels
        self.use_batchnorm = use_batchnorm
        self.base_channels = base_channels

        self.stem_block = self.get_stem_block()

        self.stage1 = self.get_stage(num_blocks_per_stage[0], 1)
        self.stage2 = self.get_stage(num_blocks_per_stage[1], 2)
        self.stage3 = self.get_stage(num_blocks_per_stage[2], 3)
        self.stage4 = self.get_stage(num_blocks_per_stage[3], 4)

        self.avg_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.dropout = nn.Dropout(dropout_value) if dropout_value is not None else nn.Identity()
        last_conv_layer_dim = self.get_num_channels(4)[2]
        self.fc = nn.Linear(2 * 2 * 2 * last_conv_layer_dim, num_classes)

    def get_stem_block(self) -> nn.Sequential:
        """
        Get the initial convolutional block (7x7x7) and the pooling layer (3x3x3) of the network.
        """
        return nn.Sequential(
            next(self.block)(self.input_channels, self.base_channels, self.base_channels, kernel_size=7, stride=2,
                             bias=False, use_batchnorm=self.use_batchnorm),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )

    def get_stage(self, num_blocks: int, stage_number: int) -> nn.Sequential:
        """
        Get a stage (stack of convolutional blocks) of the network. The first block in the stage has a stride of 2
        (bottleneck block), while the rest have a stride of 1 (residual blocks).

        :param num_blocks: Number of blocks in the stage.
        :param stage_number: Number of the stage. This number determines the number of input and output
        channels in the stage.
        :return: Stage of the network as a sequential module.
        """
        in_channels, inside_channels, out_channels = self.get_num_channels(stage_number)

        stage = nn.Sequential()
        for i in range(num_blocks):
            if i == 0:
                stage.add_module(f"block{i}", next(self.block)(in_channels, inside_channels, out_channels, stride=2,
                                                               use_batchnorm=self.use_batchnorm))
            else:
                stage.add_module(f"block{i}", next(self.block)(out_channels, inside_channels, out_channels,
                                                               use_batchnorm=self.use_batchnorm))
        return stage

    def _get_block_generator(self) -> cycle:
        """Get the generator of block constructors based on the block type."""
        if self.block_type == "A":
            return cycle([P3DBlockTypeA])
        elif self.block_type == "B":
            return cycle([P3DBlockTypeB])
        elif self.block_type == "C":
            return cycle([P3DBlockTypeC])
        elif self.block_type == "sequential":
            return cycle([P3DBlockTypeA, P3DBlockTypeB, P3DBlockTypeC])
        else:
            raise ValueError("Invalid block type.")

    def get_num_channels(self, stage_number: int) -> tuple[int, int, int]:
        """
        Get the number of input, inside and output channels for a stage of the network.

        in_channels are the number of channels in the output of the previous stage.
        inside_channels are the number of channels in the inside convolutional layers of the stage.
        out_channels are the number of channels in the last convolutional layer of every block in the stage.
        """
        in_channels = (self.base_channels * 2 ** stage_number) if stage_number != 1 else self.base_channels
        inside_channels = self.base_channels * 2 ** (stage_number - 1)
        out_channels = self.base_channels * 4 * 2 ** (stage_number - 1)
        return in_channels, inside_channels, out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        :param x: Input tensor of shape (batch_size, input_channels, depth, height, width).
        :return: Output tensor of shape (batch_size, num_classes).
        """
        x = self.stem_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    net = P3DResnet(3, 10, block_type="sequential")
    torchsummary.summary(net, (3, 16, 224, 224))
