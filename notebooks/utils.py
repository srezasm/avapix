import torch
import numpy as np
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt


def tensor_to_numpy(tensor: Tensor):
    return (
        tensor.multiply(255)
        .detach()
        .cpu()
        .numpy()
        .squeeze()
        .transpose((1, 2, 0))
        .astype(np.uint8)
    )

def file_to_numpy(file_name):
    img = np.array(Image.open(file_name))
    return img


def show_model_output_image(tensor: Tensor, title: str = ""):
    plt.imshow(tensor_to_numpy(tensor))
    plt.title(title)
    plt.axis("off")
    plt.show()


def numpy_to_tensor(arr: np.ndarray, device="cuda") -> Tensor:
    if arr.max() > 1:
        arr = arr / 255
    return (
        torch.tensor(
            arr,
            dtype=torch.float32,
            device=device,
        )
        .permute(2, 0, 1)
    )


def file_to_tensor(file_name, device="cuda") -> Tensor:
    img = np.array(Image.open(file_name))
    return numpy_to_tensor(img)


def img_to_tensor(img, device="cuda") -> Tensor:
    if isinstance(img, str):
        return file_to_tensor(img, device)
    elif isinstance(img, np.ndarray):
        return numpy_to_tensor(img, device)
