import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 32, 2, 1)
        self.conv4 = nn.Conv2d(32, 64, 2, 1)

        self.batchnorm1 = nn.BatchNorm2d(8)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm2d(32)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 1)

        self.batchnorm4 = nn.BatchNorm1d(64)
        self.batchnorm5 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)

        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)

        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)

        x = F.relu(self.conv4(x))

        x = self.flatten(x)

        x = F.tanh(self.linear1(x))
        
        x = F.leaky_relu(self.linear2(x))
        x = self.batchnorm4(x)

        x = F.tanh(self.linear3(x))

        x = F.leaky_relu(self.linear4(x))
        x = self.batchnorm5(x)

        x = F.sigmoid(self.linear5(x))

        return x


def findConv2dOutShape(hin, win, conv, pool=2):
    # get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor(
        (hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    wout = np.floor(
        (win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)
