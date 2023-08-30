import numpy as np
from numpy import ndarray
from PIL import Image
from typing import IO
from hashlib import sha256
from time import time
from typing import Any
import os

from avapix.common.processor import Processor
from avapix.common.configs import last_version

AVATAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))


processor = Processor()


def embed(text: str, random_seed: int = None, version: str = last_version) -> str:
    generated_img = processor.embed(text, random_seed, version)
    file_name = __save(generated_img, text)

    return file_name


def extract(img_stream: IO[bytes] | Any) -> str:
    image = Image.open(img_stream)

    if image.size[0] != image.size[1]:
        raise ValueError("Invalid image: input image must be square")

    # TODO: return warning for RGBA model
    if image.mode != "RGB":
        raise ValueError("Invalid image: input image must be RGB")
    
    image = np.array(image)
    decoded_text = processor.extract(image)

    return decoded_text


def __save(image: ndarray, text: str, upscale_factor: int = 30):
    hash_text = time().hex() + text
    file_name = f"avatar_{sha256(hash_text.encode()).hexdigest()[:10]}.png"
    image_path = os.path.join(AVATAR_DIR, file_name)

    image = __upscale(image, upscale_factor)

    Image.fromarray(image, "RGB").save(image_path)

    return file_name


def __upscale(image: ndarray, factor: int = 30) -> str:
    return np.kron(image, np.ones((factor, factor, 1))).astype(np.uint8)
