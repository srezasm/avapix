import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from settings import *
import matplotlib.pyplot as plt
import math

def gen_pixel_order_v1(random_seed: int):
    np.random.seed(random_seed)

    left_pixels = (
        np.arange(0, 64)
        .reshape(8, 8)[:, :4]
        .reshape(-1)
    )

    # shuffle left pixels
    shuffled_left_pixels = np.random.permutation(left_pixels)

    index_order = []

    # for each color channel (RGB: 0=Red, 1=Green, 2=Blue)
    for left, right in zip([2, 0, 1], [0, 1, 2]):
        for pixel in shuffled_left_pixels:
            if pixel == 0:
                row = 1
            elif pixel % 8 == 0:
                row = pixel // 8 + 1
            else:
                row = math.ceil(pixel / 8)

            first = (row - 1) * 8
            last = row * 8 - 1
            position = pixel - first
            mirror = last - position

            # append index for the current channel
            index_order.append(pixel*3 + left)

            # append index for the mirrored channel
            index_order.append(mirror*3 + right)

    # remove the first color channel to store the text length
    index_order.remove(TEXT_LENGTH_INDEX)

    # remove the down-left pixels first color channel to store random seed
    index_order.remove(RANDOM_SEED_INDEX)

    # remove the down-right pixels first color channel to store version
    index_order.remove(VERSION_NUM_INDEX)

    return index_order


def embed_raw_img_v1(text: str, random_seed: int) -> Tensor:
    max_text_len = 188
    text_length = len(text)

    if text_length > max_text_len:
        raise Exception(
            message=f'Text must be shorter than {max_text_len} characters.')

    index_order = gen_pixel_order_v1(random_seed)

    img = torch.zeros(8*8*3, dtype=torch.int)

    img[TEXT_LENGTH_INDEX] = text_length
    img[RANDOM_SEED_INDEX] = random_seed
    img[VERSION_NUM_INDEX] = V1_NUMBER

    for i, c in enumerate(text):
        img[index_order[i]] = ord(c)

    return img.reshape((8, 8, 3))


def extract_text(tensor_img: torch.Tensor):
    img = tensor_img.clone()

    img = img.squeeze()

    if img.ndim == 3:
        img = img.permute((1, 2, 0))
        img = img.reshape(-1)
    if img.max() <= 1.0:
        img *= 255
        img = img.to(torch.uint8)

    version_num = img[VERSION_NUM_INDEX]
    if version_num == V1_NUMBER:
        text_length = img[TEXT_LENGTH_INDEX].item()
        random_seed = img[RANDOM_SEED_INDEX].item()
        index_order = gen_pixel_order_v1(random_seed)

    text = ''

    for i in range(text_length):
        text += chr(img[index_order[i]].item())

    return text


def img_tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy().squeeze().transpose((1, 2, 0))

def get_original_size(upscaled: ndarray):
    if upscaled.shape[0] != upscaled.shape[1]:
        raise Exception('Invalid image: input image must be square')
    
    size = upscaled.shape[0]
    upscale_factor = upscaled.shape[0] // 8

    rows, cols = np.meshgrid(np.arange(0, size, upscale_factor), np.arange(0, size, upscale_factor))
    pixel_locs = np.column_stack((rows.ravel(), cols.ravel()))

    orig_img = np.zeros((8, 8, 3))
    for i, j in pixel_locs:
        orig_img[i//upscale_factor, j//upscale_factor] = upscaled[i, j]
