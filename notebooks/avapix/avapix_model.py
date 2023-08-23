import torch
import torch.nn as nn
import torch.nn.functional as F


# class AvapixModel(nn.Module):
#     def __init__(self, device) -> None:
#         super().__init__()

#         self.conv01 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)
#         self.conv02 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2)
#         self.conv03 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
#         self.conv04 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)

#         self.batch_norm = nn.BatchNorm2d(64)

#         self.conv07 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4)
#         self.conv08 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
#         self.conv09 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=2)
#         self.conv10 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1)

#         self.to(device)

#     def forward(self, x):
#         x = F.relu(self.conv01(x))
#         x = F.relu(self.conv02(x))
#         x = F.relu(self.conv03(x))
#         x = F.relu(self.conv04(x))
        
#         x = self.batch_norm(x)
        
#         x = F.relu(self.conv07(x))
#         x = F.relu(self.conv08(x))
#         x = F.relu(self.conv09(x))

#         x = self.conv10(x)

#         x = x.squeeze()
        
#         return x

class AvapixModel(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.conv01 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)
        self.conv02 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1)
        self.conv03 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1)
        self.conv04 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)

        self.batch_norm = nn.BatchNorm2d(64)

        self.conv07 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv08 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
        self.conv09 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        
        x = self.batch_norm(x)
        
        x = F.relu(self.conv07(x))
        x = F.relu(self.conv08(x))
        x = F.relu(self.conv09(x))
        x = F.sigmoid(self.conv10(x))

        return x