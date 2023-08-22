import torch.nn.functional as F
import torch.nn as nn
from settings import DEVICE


class FaceValidateV1_1(nn.Module):
    def __init__(self):
        super(FaceValidateV1_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)

        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features=64, out_features=32)
        self.linear2 = nn.Linear(in_features=32, out_features=16)
        self.linear3 = nn.Linear(in_features=16, out_features=8)
        self.linear4 = nn.Linear(in_features=8, out_features=1)

        self.sigmoid = nn.Sigmoid()

        self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = self.batchnorm1(x)
        
        x = F.relu(self.conv3(x))
        x = self.batchnorm2(x)
        
        x = F.relu(self.conv4(x))
        x = self.batchnorm3(x)
        
        x = self.flatten(x)
        
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))

        x = self.sigmoid(x)

        return x
