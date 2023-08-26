from PIL import Image
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor


def load_from_disk(file_name):
    return Image.open(file_name)


def tensor_to_numpy(tensor_img: Tensor) -> ndarray:
    return tensor_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()


def numpy_to_tensor(np_img: ndarray, device: str = None) -> Tensor:
    tensor_img = torch.tensor(np_img/255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)

    if device is not None:
        tensor_img = tensor_img.to(device)

    return tensor_img

def load_from_request(file_stream) -> Image:
    return Image.open(file_stream)

def load_from_file(file_name) -> Image:
    return Image.open(file_name)

def image_to_numpy(image: Image) -> ndarray:
    return np.array(image)

def get_original_size(upscaled: ndarray) -> ndarray:
    if upscaled.shape[0] != upscaled.shape[1]:
        raise ValueError('Invalid image: input image must be square')
    
    size = upscaled.shape[0]
    upscale_factor = size // 8

    rows, cols = np.meshgrid(np.arange(0, size, upscale_factor), np.arange(0, size, upscale_factor))
    pixel_locs = np.column_stack((rows.ravel(), cols.ravel()))

    orig_img = np.zeros((8, 8, 3))
    for i, j in pixel_locs:
        orig_img[i//upscale_factor, j//upscale_factor] = upscaled[i, j]

    return orig_img

