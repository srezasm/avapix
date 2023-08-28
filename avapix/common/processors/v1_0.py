from avapix.common.processors.base_processor import BaseProcessor
from avapix.common.constants import *

from numpy import ndarray
import numpy as np
import math


class ProcessorV1_0(BaseProcessor):
    def __init__(self) -> None:
        super().__init__()

        self.version_num = 100
        self.max_text_len = 188

    def get_version_num(self) -> int:
        return self.version_num

    def __gen_pixel_order__(self, random_seed: int) -> ndarray:
        '''
        Symmetric color channel index generator

        Parameters
        ----------
        random_seed : int
            Random seed to enable reproducibility
        '''
        np.random.seed(random_seed)

        left_pixels = np.arange(0, 64).reshape(8, 8)[:, :4].reshape(-1)

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

                index_order.extend([pixel * 3 + left, mirror * 3 + right])

        # remove the indices for metadata channels
        metadata_indices = [
            self.TEXT_LENGTH_INDEX,
            self.RANDOM_SEED_INDEX,
            self.VERSION_NUM_INDEX,
        ]
        index_order = [index for index in index_order if index not in metadata_indices]

        return index_order

    def embed(self, text, random_seed: int) -> ndarray:
        '''
        Embeds text into a raw RGB image array.

        Parameters
        ----------
        text : str
            Text to embed into the image.
        random_seed : int, optional
            Random seed to use for generating the embedding order, by default 42

        Returns
        -------
        ndarray
            Standard RGB image array with shape (8, 8, 3).
        '''

        text_length = len(text)

        if text_length > self.max_text_len:
            raise Exception(
                message=f"Text must be shorter than {self.max_text_len} characters."
            )

        if any(ord(text[i]) < 0 or ord(text[i]) > 254 for i in range(len(text))):
            raise Exception("Text ascii code must be in inclusive range of [0, 254]")

        index_order = self.__gen_pixel_order__(random_seed)

        img = np.zeros(8 * 8 * 3, dtype=np.uint8)

        img[self.TEXT_LENGTH_INDEX] = text_length
        img[self.RANDOM_SEED_INDEX] = random_seed
        img[self.VERSION_NUM_INDEX] = self.version_num

        img[index_order[:text_length]] = [ord(c) for c in text]

        return img.reshape((8, 8, 3))

    def extract(self, image_array: ndarray) -> str:
        '''
        Extracts text from an image array.

        Parameters
        ----------
        image_array : ndarray
            Standard RGB image array with shape (8, 8, 3).

        Returns
        -------
        str
            Extracted text from the image.
        '''

        img = image_array.squeeze()

        if img.ndim != 3:
            raise Exception("Image must have 3 dimensions.")
        if img.shape != (8, 8, 3):
            raise Exception("Image must be in shape (8, 8, 3)")
        if img.max() <= 1.0:
            raise Exception("Image must be in range [0, 255]")

        img_flat = img.reshape(-1)

        version_num = img_flat[self.VERSION_NUM_INDEX]

        if version_num != self.version_num:
            raise Exception(f"Image version number must be {self.version_num}.")

        text_length = img_flat[self.TEXT_LENGTH_INDEX]
        random_seed = img_flat[self.RANDOM_SEED_INDEX]
        index_order = self.__gen_pixel_order__(random_seed)

        text = "".join([chr(img_flat[index]) for index in index_order[:text_length]])
        return text
