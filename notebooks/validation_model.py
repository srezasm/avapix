import torch
import torch.nn as nn
import torch.functional as F


class ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 2, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.Tanh(),
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Sequential( 
            nn.Conv2d(32, 64, 4),
            nn.Tanh(),
            nn.BatchNorm2d(64)
        )

        self.flatten = nn.Flatten()

        self.sigmoid = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        device = x.device
        self.to(device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.sigmoid(x)

        return x
