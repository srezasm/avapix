import torch
import torch.nn as nn
import torch.nn.functional as F


class ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # cnn
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 2),  # 7x7
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3),  # 4x4
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),  # 2x2
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 16, 2, 2),  # 4x4
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),  # 2x2
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 16, 4, 4), nn.ReLU(), nn.BatchNorm2d(16)  # 2x2
        )

        self.ln1 = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Tanh(),
            nn.BatchNorm1d(2),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # cnn
        img1 = self.conv1(img)  # [16, 16, 2, 2]
        img2 = self.conv2(img)  # [16, 16, 2, 2]
        img3 = self.conv3(img)  # [16, 16, 2, 2]

        img = torch.concat((img1, img2, img3), dim=1)  # concat
        img = img.reshape(img.shape[0], -1)  # flatten

        img = self.ln1(img)

        return img
