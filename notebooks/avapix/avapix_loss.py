import torch
import torch.nn as nn


class AvapixLoss(nn.Module):
    def __init__(self, zero_limit_value: float = 0.01) -> None:
        super().__init__()

        self.zero_limit_value = zero_limit_value

    def forward(self, x: torch.Tensor):
        device = x.device

        x = x.masked_fill(x < self.zero_limit_value, self.zero_limit_value)

        one_tensor = torch.tensor([1.0], device=device)
        loss = torch.tan(one_tensor) / x - torch.tan(x)
        loss = loss.mean()

        return loss.requires_grad_(True)
