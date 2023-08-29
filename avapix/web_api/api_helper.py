import numpy as np
from numpy import ndarray
from PIL import Image
from typing import IO
from hashlib import sha256
from time import time
from typing import Any
import os

from avapix.common.processor_wrapper import ProcessorWrapper
from avapix.web_api.api_constants import AVATAR_DIR


class Helper:
    def __init__(self) -> None:
        self.processor = ProcessorWrapper()

    def embed(self, text: str, random_seed: int = None) -> str:
        generated_img = self.processor.embed(text, random_seed)
        file_name = self.save(generated_img, text)

        return file_name

    def save(self, image: ndarray, text: str, upscale_factor: int = 30):
        hash_text = time().hex() + text
        file_name = f"avatar_{sha256(hash_text.encode()).hexdigest()[:10]}.png"
        image_path = os.path.join(AVATAR_DIR, file_name)

        image = self.upscale(image, upscale_factor)

        Image.fromarray(image, "RGB").save(image_path)

        return file_name

    def upscale(self, image: ndarray, factor: int = 30):
        return np.kron(image, np.ones((factor, factor, 1))).astype(np.uint8)

    def extract(self, img_stream: IO[bytes] | Any) -> str:
        decoded_text = self.processor.extract(img_stream)

        return decoded_text
