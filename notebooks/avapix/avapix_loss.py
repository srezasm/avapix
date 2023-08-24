import torch
import torch.nn as nn


class AvapixLoss(nn.Module):
    def __init__(self, zero_limit_value: float = 0.01) -> None:
        super().__init__()

        self.zero_limit_value = zero_limit_value

    def __calc_face_shape_loss__(self, x: torch.Tensor):
        x = x.masked_fill(x < self.zero_limit_value, self.zero_limit_value)

        one_tensor = torch.tensor([1.0], device=self.device)
        loss = torch.tan(one_tensor) / x - torch.tan(x)
        loss = loss.pow(2).mean()

        return loss

    def __calc_diff_loss__(self, output_pic: torch.Tensor, org_pic: torch.Tensor):
        mask = org_pic != 0

        result = torch.zeros_like(output_pic)
        result[mask] = output_pic[mask]

        loss = (result - org_pic).pow(2).mean()

        return loss

    def forward(self, x: torch.Tensor, output_pic: torch.Tensor, org_pic: torch.Tensor):
        self.device = x.device

        loss = self.__calc_face_shape_loss__(x)
        loss += self.__calc_diff_loss__(output_pic, org_pic)

        return loss.requires_grad_(True)
