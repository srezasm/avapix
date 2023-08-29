import os
import torch
import torch.nn as nn


class V1(nn.Module):
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
            nn.BatchNorm2d(64),
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
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=1),
        )

        self.output = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        device = x.device
        self.to(device)

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)

        return x

    def load_checkpoint(self, device):
        checkpoint_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'checkpoints/v1.pth')
        )

        self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.eval()

    @torch.no_grad()
    def gen(self, x: torch.Tensor):
        return self(x)
