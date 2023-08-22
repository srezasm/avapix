import torch
import torch.nn as nn
import torch.nn.functional as F


class AvapixModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv01 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, padding=0)
        self.conv02 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding=0)
        self.conv03 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv04 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=0)


        # self.conv05 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=8, padding=0)
        self.batch_norm = nn.BatchNorm2d(64)
        # self.conv06 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=8, padding=0)

        self.conv07 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, padding=0)
        self.conv08 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=0)
        self.conv09 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=2, padding=0)
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        
        # x = F.relu(self.conv05(x))
        x = self.batch_norm(x)
        # x = F.relu(self.conv06(x))
        
        x = F.relu(self.conv07(x))
        x = F.relu(self.conv08(x))
        x = F.relu(self.conv09(x))

        x = self.conv10(x)

        x = x.squeeze()
        
        return x