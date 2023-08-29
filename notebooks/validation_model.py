import torch
import torch.nn as nn
import torch.functional as F


class ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1_encode = nn.Sequential(
            nn.Conv2d(3, 6, 1),
            nn.LeakyReLU(),

            nn.Conv2d(6, 8, 2),
            nn.LeakyReLU(),

            nn.Conv2d(8, 16, 4),
            nn.LeakyReLU()
        )

        self.conv2_decode = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 6, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(6, 3, 1),
            nn.ReLU()
        )

        self.conv3_sym = nn.Sequential(
            nn.Conv2d(3, 32, (8, 4)),
            nn.ReLU(),
            nn.Tanh(),
        )

        self.flatten = nn.Flatten()

        self.sigmoid = nn.Sequential(
            nn.Linear(in_features=160, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        device = x.device
        self.to(device)

        x = self.conv1_encode(x)
        x = self.conv2_decode(x)
        x = self.conv3_sym(x)

        x = self.flatten(x)

        x = self.sigmoid(x)

        return x
