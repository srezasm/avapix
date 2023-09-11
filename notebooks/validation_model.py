import torch
import torch.nn as nn


class ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        # BatchNorm
        self.bn = nn.BatchNorm1d(256)

        # Linear
        self.ln1 = nn.Linear(256, 512)
        self.do1 = nn.Dropout(0.5)

        self.ln2 = nn.Linear(512, 1024)
        self.do2 = nn.Dropout(0.8)

        self.ln3 = nn.Linear(1024, 512)
        self.do3 = nn.Dropout(0.8)

        self.ln4 = nn.Linear(512, 1)

    def forward(self, img):
        img = self.encode(img)

        # Flatten
        img = img.reshape(img.shape[0], -1)

        # Batch norm
        img = self.bn(img)

        # Linear
        img = torch.relu(self.ln1(img))
        img = self.do1(img)

        img = torch.relu(self.ln2(img))
        img = self.do2(img)

        img = torch.relu(self.ln3(img))
        img = self.do3(img)

        img = torch.sigmoid(self.ln4(img))

        return img
