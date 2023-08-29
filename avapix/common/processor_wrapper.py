import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from PIL import Image
from typing import IO
from hashlib import sha256
from time import time

from avapix.common.processor import Processor


class ProcessorWrapper:
    '''
    Wrapper around Processor to provide common interface for
    embedding and extracting text from images in CLI and web API
    projects
    '''
    def __init__(self) -> None:
        self.processor = Processor()

    def embed(self, text, random_seed: int = None) -> ndarray:
        generated_img = self.processor.embed(text, random_seed)

        return generated_img
    
    def extract(self, img_file) -> str:
        numpy_img = self.__to_original__(img_file)
        decoded_text = self.processor.extract(numpy_img)

        return decoded_text

    def __to_original__(self, img_file) -> ndarray:
        image = Image.open(img_file)

        if image.size[0] != image.size[1]:
            raise ValueError("Invalid image: input image must be square")

        # TODO: return warning for RGBA model
        if image.mode != "RGB":
            raise ValueError("Invalid image: input image must be RGB")

        size = image.size[0]
        upscale_factor = size // 8

        rows, cols = np.meshgrid(
            np.arange(0, size, upscale_factor), np.arange(0, size, upscale_factor)
        )
        pixel_locs = np.column_stack((rows.ravel(), cols.ravel()))

        orig_img = np.zeros((8, 8, 3), np.uint8)
        for i, j in pixel_locs:
            orig_img[i // upscale_factor, j // upscale_factor] = image.getpixel((j, i))

        return orig_img
