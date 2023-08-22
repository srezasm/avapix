import torch
import torch.nn as nn
import numpy as np

class AvapixLoss(nn.Module):
    def forward(self, x):
        loss = np.tan(1) / x - np.tan(x)
        # loss = torch.tan(torch.tensor([1])) / x - torch.tan(x)
        return torch.tensor(loss)