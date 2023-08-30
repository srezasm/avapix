import numpy as np
from numpy import ndarray
from PIL import Image
from typing import IO
from typing import Any
import os

from avapix.common.processor import Processor

AVATAR_DIR = os.path.join(".", "avatars")

if not os.path.isdir(AVATAR_DIR):
    os.mkdir(AVATAR_DIR)


processor = Processor()


def embed(
    text: str,
    export_sizes: list[int],
    random_seed,
    version: str,
) -> list[str]:
    if random_seed == "":
        random_seed = None
    else:
        random_seed = int(random_seed)

    generated_img = processor.embed(text, random_seed, version)

    file_names = __save(generated_img, export_sizes)

    return file_names


def extract(img_path: str) -> str:
    image = Image.open(img_path)

    if image.size[0] != image.size[1]:
        raise ValueError("Invalid image: input image must be square")

    # TODO: return warning for RGBA model
    if image.mode != "RGB":
        raise ValueError("Invalid image: input image must be RGB")

    image = np.array(image)
    decoded_text = processor.extract(image)

    return decoded_text


def __save(image: ndarray, sizes: list[int]) -> list[str]:
    file_names = []
    for size in sizes:
        file_name = f"avatar_{size}.png"
        image_path = os.path.join(AVATAR_DIR, file_name)

        upscaled = __upscale(image, size // 8)

        Image.fromarray(upscaled, "RGB").save(image_path)

        file_names.append(image_path)

    return file_names


def __upscale(image: ndarray, factor: int):
    return np.kron(image, np.ones((factor, factor, 1))).astype(np.uint8)
