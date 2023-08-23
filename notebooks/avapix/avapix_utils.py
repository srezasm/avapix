import numpy as np
from numpy import ndarray
from torch import Tensor
from settings import *
import math

DEFAULT_PIXEL_ORDER = [173, 186, 83, 84, 146, 165, 101, 114, 50, 69,
                       53, 66, 176, 183, 149, 162, 74, 93, 21, 26,
                       45, 98, 117, 29, 42, 77, 90, 59, 60, 131,
                       132, 5, 18, 8, 15, 152, 159, 11, 12, 125,
                       138, 155, 156, 128, 135, 104, 111, 179, 180, 122,
                       141, 35, 36, 56, 63, 80, 87, 107, 108, 32,
                       39, 171, 187, 81, 85, 144, 166, 99, 115, 48,
                       70, 51, 67, 174, 184, 147, 163, 72, 94, 0,
                       22, 24, 46, 96, 118, 27, 43, 75, 91, 57,
                       61, 129, 133, 3, 19, 6, 16, 150, 160, 9,
                       13, 123, 139, 153, 157, 126, 136, 102, 112, 177,
                       181, 120, 142, 33, 37, 54, 64, 78, 88, 168,
                       190, 105, 109, 30, 40, 172, 188, 82, 86, 145,
                       167, 100, 116, 49, 71, 52, 68, 175, 185, 148,
                       164, 73, 95, 1, 23, 25, 47, 97, 119, 28,
                       44, 76, 92, 58, 62, 130, 134, 4, 20, 7,
                       17, 151, 161, 10, 14, 124, 140, 154, 158, 127,
                       137, 103, 113, 178, 182, 121, 143, 34, 38, 55,
                       65, 79, 89, 169, 191, 106, 110, 31, 41]


def gen_pixel_order_v1(random_seed: int):
    if random_seed == DEFAULT_RANDOM_SEED:
        return DEFAULT_PIXEL_ORDER
    
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


def generate_input_v1(img: Tensor, random_seed: int,
                      text_length: int) -> Tensor:
    index_order = gen_pixel_order_v1(random_seed)

    result_img = img.clone().reshape(-1)

    # embed metadata
    result_img[TEXT_LENGTH_INDEX] = text_length
    result_img[RANDOM_SEED_INDEX] = random_seed
    result_img[VERSION_NUM_INDEX] = 100

    excess_indexes = index_order[text_length:]
    result_img[excess_indexes] = 0

    return result_img
