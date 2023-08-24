import numpy as np
from numpy import ndarray
from torch import Tensor
from settings import *
import matplotlib.pyplot as plt
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


def img_tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy().squeeze().transpose((1, 2, 0))


def show_model_output_image(tensor: torch.Tensor, title: str = ''):
    plt.imshow(img_tensor_to_numpy(tensor))
    plt.title(title)
    plt.axis('off')
    plt.show()


def CIEDE2000(Lab_1, Lab_2):
    '''
    Calculates CIEDE2000 color distance between two CIE L*a*b* colors
    Source: https://github.com/lovro-i/CIEDE2000
    '''

    C_25_7 = 6103515625  # 25**7

    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))

    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = math.sqrt(a1_**2 + b1_**2)
    C2_ = math.sqrt(a2_**2 + b2_**2)

    if b1_ == 0 and a1_ == 0:
        h1_ = 0
    elif a1_ >= 0:
        h1_ = math.atan2(b1_, a1_)
    else:
        h1_ = math.atan2(b1_, a1_) + 2 * math.pi

    if b2_ == 0 and a2_ == 0:
        h2_ = 0
    elif a2_ >= 0:
        h2_ = math.atan2(b2_, a2_)
    else:
        h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0:
        dh_ = 0
    elif dh_ > math.pi:
        dh_ -= 2 * math.pi
    elif dh_ < -math.pi:
        dh_ += 2 * math.pi
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)

    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2

    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    if _dh <= math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2
    elif _dh > math.pi and _sh < 2 * math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh > math.pi and _sh >= 2 * math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2 - math.pi
    else:
        h_ave = h1_ + h2_

    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * \
        math.cos(3 * h_ave + math.pi / 30) - 0.2 * \
        math.cos(4 * h_ave - 63 * math.pi / 180)

    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0:
        h_ave_deg += 360
    elif h_ave_deg > 360:
        h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))

    R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50)**2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1

    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H

    dE_00 = math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    return dE_00
