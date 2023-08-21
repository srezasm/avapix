import torch
import torch.nn as nn


class FaceValidateV1(nn.Module):
    def __init__(self):
        super(FaceValidateV1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, padding=0)
        
        self.dropout1 = nn.Dropout2d(p=0.4)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.dropout3 = nn.Dropout2d(p=0.1)
        
        self.flatten = nn.Flatten()
        
        self.tanh1 = nn.Linear(in_features=8*8, out_features=64)
        self.tanh2 = nn.Linear(in_features=64, out_features=16)
        self.tanh3 = nn.Linear(in_features=16, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)

        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)

        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)

        x = self.flatten(x)

        x = torch.tanh(self.tanh1(x))
        x = torch.tanh(self.tanh2(x))
        x = torch.tanh(self.tanh3(x))

        x = self.sigmoid(x)
        
        return x