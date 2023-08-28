import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from PIL import Image
from typing import IO
from hashlib import sha256
from time import time

from avapix.common.models.avapix_model import AvapixModel
from avapix.common.constants import *
from avapix.common.processor import Processor
from avapix.web_api.api_constants import *

processor = Processor()

class EmbedWrapper:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AvapixModel()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    # TODO: optionally get random seed from user
    # TODO: optionally get version from user
    def embed(self, text) -> str:
        random_seed = np.random.randint(0, 256)
        raw_embed = processor.embed(text, random_seed)
        generated_img = self.__gen__(raw_embed)
        file_name = self.__save__(generated_img, text)

        return file_name

    @torch.no_grad()
    def __gen__(self, raw_embed_img: ndarray):
        result = self.model(self.__to_tensor__(raw_embed_img))
        result = self.__to_numpy__(result)

        # replace the originally embedded color channels
        mask = raw_embed_img != 0
        result[mask] = raw_embed_img[mask]

        return result

    def __to_tensor__(self, img_arr: ndarray) -> Tensor:
        return (
            torch.tensor(img_arr / 255, dtype=torch.float, device=self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

    def __to_numpy__(self, img_tensor: Tensor) -> ndarray:
        return (
            img_tensor.detach()
            .cpu()
            .multiply(255)
            .numpy()
            .squeeze()
            .transpose(1, 2, 0)
            .astype(np.uint8)
        )

    def __save__(self, image: ndarray, text: str, upscale_factor: int = 30):
        hash_text = time().hex() + text
        file_name = f"avatar_{sha256(hash_text.encode()).hexdigest()[:10]}.png"
        image_path = os.path.join(AVATAR_DIR, file_name)

        image = self.__upscale__(image, upscale_factor)

        Image.fromarray(image, "RGB").save(image_path)

        return file_name

    def __upscale__(self, image: ndarray, factor: int = 30):
        return np.kron(image, np.ones((factor, factor, 1))).astype(np.uint8)


class DecodeWrapper:
    def extract(self, img_stream: IO[bytes]) -> str:
        numpy_img = self.__to_original__(img_stream)
        decoded_text = processor.extract(numpy_img)
        return decoded_text

    def __to_original__(self, img_stream: IO[bytes]) -> ndarray:
        image = Image.open(img_stream)

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