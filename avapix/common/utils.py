import torch
from torch import Tensor
import numpy as np
from numpy import ndarray


def to_tensor(img_arr: ndarray, device) -> Tensor:
    return (
        torch.tensor(img_arr / 255, dtype=torch.float, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def to_numpy(img_tensor: Tensor) -> ndarray:
    return (
        img_tensor.detach()
        .cpu()
        .multiply(255)
        .numpy()
        .squeeze()
        .transpose(1, 2, 0)
        .astype(np.uint8)
    )
