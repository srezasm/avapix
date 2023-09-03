import torch
import torch.nn as nn
import torch.nn.functional as F


class AvapixModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encode1 = nn.Sequential(
            nn.Conv2d(3, 8, 2),  # 7x7
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),  # 4x4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 2x2
            nn.Tanh(),
            nn.BatchNorm2d(16),
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(3, 16, 2, 2),  # 4x4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 2x2
            nn.Tanh(),
            nn.BatchNorm2d(16),
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(3, 16, 4, 4),
            nn.Tanh(),
            nn.BatchNorm2d(16)  # 2x2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(48, 64, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 1),
            nn.Tanh(),

            nn.Conv2d(128, 256, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, 2),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4),
            nn.Tanh(),
            nn.BatchNorm2d(3)
        )

    def forward(self, img):
        img1 = self.encode1(img)
        img2 = self.encode2(img)
        img3 = self.encode3(img)

        img = torch.concat((img1, img2, img3), dim=1)

        img = self.conv(img)

        img = self.decoder(img)

        return img
