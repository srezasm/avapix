import torch
import numpy as np
from torch import Tensor
import math
import matplotlib.pyplot as plt


def img_tensor_to_numpy(tensor: Tensor):
    return (
        tensor.multiply(255)
        .detach()
        .cpu()
        .numpy()
        .squeeze()
        .transpose((1, 2, 0))
        .astype(np.uint8)
    )


def show_model_output_image(tensor: Tensor, title: str = ""):
    plt.imshow(img_tensor_to_numpy(tensor))
    plt.title(title)
    plt.axis("off")
    plt.show()

