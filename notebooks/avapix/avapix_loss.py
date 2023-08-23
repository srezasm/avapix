import torch
import torch.nn as nn
import numpy as np

# class AvapixLoss(nn.Module):
#     def __init__(self, device) -> None:
#         super().__init__()

#         self.one_tensor = torch.tensor([1],
#                                        device=device,
#                                        requires_grad=True)
#         self.device = device

#     def forward(self, x):
#         loss = torch.tan(self.one_tensor) / x - torch.tan(x)
#         return torch.tensor(loss, device=self.device)
    
class AvapixLoss(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.one_tensor = torch.tensor([1.0],
                                       device=device)
        self.device = device

    def forward(self, x):
        loss = torch.tan(self.one_tensor) / x - torch.tan(x)
        loss = loss.mean()
        return loss.to(self.device).requires_grad_(True)
