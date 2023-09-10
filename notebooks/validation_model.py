import torch
import torch.nn as nn
import torch.nn.functional as F


class ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 256, 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        )

        # CNN
        self.conv1 = nn.Conv2d(3, 16, (4, 8))
        self.mp1 = nn.MaxPool2d((2, 1))

        self.conv2 = nn.Conv2d(3, 16, (8, 4))
        self.mp2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(3, 16, 4)
        self.mp3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(16, 32, 2)

        # Linear
        self.bn = nn.BatchNorm1d(96)

        self.ln1 = nn.Linear(96, 256)
        self.do1 = nn.Dropout(0.5)

        self.ln2 = nn.Linear(256, 512)
        self.do2 = nn.Dropout(0.5)

        self.ln3 = nn.Linear(512, 256)
        self.do3 = nn.Dropout(0.5)

        self.ln4 = nn.Linear(256, 128)
        self.do4 = nn.Dropout(0.5)

        self.ln5 = nn.Linear(128, 64)
        self.do5 = nn.Dropout(0.5)

        self.ln6 = nn.Linear(64, 32)
        self.do6 = nn.Dropout(0.5)

        self.ln7 = nn.Linear(32, 1)

    def forward(self, img):
        img = self.enc(img)

        # CNN
        img1 = torch.relu(self.conv1(img))
        img1 = self.mp1(img1)
        img1 = img1.reshape(img1.shape[0], -1)

        img2 = torch.relu(self.conv2(img))
        img2 = self.mp2(img2)
        img2 = img2.reshape(img2.shape[0], -1)

        img3 = torch.relu(self.conv3(img))
        img3 = self.mp3(img3)
        img3 = torch.relu(self.conv4(img3))
        img3 = img3.reshape(img3.shape[0], -1)

        # Concatenate
        img = torch.cat([img1, img2, img3], dim=1)

        # Batch norm
        img = self.bn(img)

        # Linear
        img = torch.tanh(self.ln1(img))
        img = self.do1(img)

        img = torch.relu(self.ln2(img))
        img = self.do2(img)

        img = torch.tanh(self.ln3(img))
        img = self.do3(img)

        img = torch.relu(self.ln4(img))
        img = self.do4(img)

        img = torch.tanh(self.ln5(img))
        img = self.do5(img)

        img = torch.relu(self.ln6(img))
        img = self.do6(img)

        img = torch.sigmoid(self.ln7(img))

        return img
