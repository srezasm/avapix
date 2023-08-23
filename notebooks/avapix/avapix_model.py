import torch
import torch.nn as nn
import torch.nn.functional as F


class AvapixModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4),
            nn.Sigmoid(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.Sigmoid(),
            nn.BatchNorm2d(16),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2),
            nn.Sigmoid(),
            nn.BatchNorm2d(8),

            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=1)
        )


        self.output = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        device = x.device
        self.to(device)

        # length = x[:, 2, 0, 0]
        # seed = x[:, 2, 7, 0]
        # version = x[:, 0, 7, 7]

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)

        # x[:, 2, 0, 0] = length
        # x[:, 2, 7, 0] = seed
        # x[:, 0, 7, 7] = version

        return x
