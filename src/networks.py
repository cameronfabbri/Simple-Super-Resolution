"""
Modules and networks
"""
from typing import Callable

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    """
    Wrapper around the Convolution class to provide normalization and
    activation functions in one call.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            activation: Callable,
            normalization: Callable):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        # Set normalization and activation function
        if normalization is not None:
            self.normalization = normalization(out_channels)
        else:
            self.normalization = nn.Identity()
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass

        Params
        ------
        x (torch.Tensor): input

        Returns
        -------
        x (torch.Tensor): output
        """
        x = self.conv(x)
        x = self.normalization(x)
        return self.activation(x)


class ResBlock(nn.Module):
    """
    Basic Residual block
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size):
        """
        Params
        ------
        in_channels (int): Number of channels coming in
        out_channels (int): Number of channels for the output
        kernel_size (int): Size of the kernel

        Returns
        -------
        x (torch.Tensor): output
        """
        super(ResBlock, self).__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.PReLU,
            normalization=nn.BatchNorm2d)

        self.conv2 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.Identity,
            normalization=nn.BatchNorm2d)

    def _residual(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        return torch.nn.functional.relu(x + self._residual(x))


class SISR_Resblocks(nn.Module):
    """ Wrapper to create a Sequential module of n resblocks """
    def __init__(self, num_blocks):
        super(SISR_Resblocks, self).__init__()

        self.resblocks = []
        for i in range(num_blocks):
            self.resblocks.append(
                ResBlock(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3))
        self.resblocks = nn.Sequential(*self.resblocks)

    def forward(self, x):
        return self.resblocks(x)


class Generator(nn.Module):
    """ Class for the Generator based on SRGAN """
    def __init__(self, resblocks):
        super(Generator, self).__init__()

        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=9,
            stride=1,
            padding=9//2,
            activation=nn.PReLU,
            normalization=None)

        self.conv2 = Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.Identity,
            normalization=nn.BatchNorm2d)

        self.resblocks = resblocks

        self.conv3 = Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.Identity,
            normalization=None)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.prelu = nn.PReLU()

        self.conv4 = Conv2d(
            in_channels=16,
            out_channels=3,
            kernel_size=9,
            stride=1,
            padding=9//2,
            activation=nn.Tanh,
            normalization=None)

    def forward(self, x):
        skip = self.conv1(x)
        x = self.resblocks(skip)
        x = self.conv2(x) + skip
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        x = self.pixel_shuffle(x)
        x = self.conv4(x)
        return x


class Discriminator(nn.Module):
    """ Class for the Discriminator based on SRGAN """
    def __init__(self, fc_size):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=3//2,
            activation=nn.LeakyReLU,
            normalization=None)

        self.conv2 = Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=3//2,
            activation=nn.LeakyReLU,
            normalization=nn.BatchNorm2d)

        self.conv3 = Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=3//2,
            activation=nn.LeakyReLU,
            normalization=nn.BatchNorm2d)

        self.conv4 = Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=3//2,
            activation=nn.LeakyReLU,
            normalization=nn.BatchNorm2d)

        self.conv5 = Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=3//2,
            activation=nn.LeakyReLU,
            normalization=nn.BatchNorm2d)

        self.conv6 = Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=3//2,
            activation=nn.LeakyReLU,
            normalization=nn.BatchNorm2d)

        self.conv7 = Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=3//2,
            activation=nn.LeakyReLU,
            normalization=nn.BatchNorm2d)

        self.conv8 = Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=3//2,
            activation=nn.LeakyReLU,
            normalization=nn.BatchNorm2d)

        self.fc1 = nn.Linear(
            in_features=fc_size,
            out_features=1024)

        self.fc2 = nn.Linear(
            in_features=1024,
            out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        # Flatten and put through a fc layer
        s = (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = torch.reshape(x, s)
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        return self.fc2(x)

