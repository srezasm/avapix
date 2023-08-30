import torch
from torch import Tensor
import numpy as np
from numpy import ndarray

from avapix.common.processors.base_processor import VERSION_NUM_INDEX
from avapix.common.processors.v1_0 import ProcessorV1_0
from avapix.common.processors.v1_1 import ProcessorV1_1
from avapix.common.models.avapix_models import V1
from avapix.common.configs import last_version

class Processor:
    """
    Implemented with Factory, and Strategy patterns to manage
    different model and processor versions
    """

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Cache the latest model version and load the previous versions
        # on demand
        self.model = V1()
        self.model.load_checkpoint(self.device)

    def embed(
        self,
        text: str,
        random_seed: int = None,
        version: str = last_version,
    ):
        processor = None

        if version == "v1.0":
            processor = ProcessorV1_0()
        elif version == "v1.1":
            processor = ProcessorV1_1()
        else:
            raise ValueError(f"Invalid processor version: {version}")

        raw_embedded_img = processor.embed(text, random_seed)

        # Model version is not bound to the Processor version, e.g. there
        # might  be a change in the  embedding  algorithm, but  the model
        # architecture remains the same.

        if version in ["v1.0", "v1.1"]:
            gen_result = self.model.gen(self.__to_tensor(raw_embedded_img, self.device))

        gen_result = self.__to_numpy(gen_result)

        # Replace the originally embedded color channels
        mask = raw_embedded_img != 0
        gen_result[mask] = raw_embedded_img[mask]

        return gen_result

    def extract(self, image_array: ndarray):
        orig_img = self.__to_original(image_array)

        version = orig_img.reshape(-1)[VERSION_NUM_INDEX]

        processor = None
        if version == 100:
            processor = ProcessorV1_0()
        elif version == 110:
            processor = ProcessorV1_1()
        else:
            raise ValueError(f"Invalid version: {version}")

        return processor.extract(orig_img)

    def __to_tensor(self, img_arr: ndarray, device) -> Tensor:
        return (
            torch.tensor(img_arr / 255, dtype=torch.float, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )


    def __to_numpy(self, img_tensor: Tensor) -> ndarray:
        return (
            img_tensor.detach()
            .cpu()
            .multiply(255)
            .numpy()
            .squeeze()
            .transpose(1, 2, 0)
            .astype(np.uint8)
        )
    
    def __to_original(self, image: ndarray) -> ndarray:
        size = image.shape[0]
        upscale_factor = size // 8

        rows, cols = np.meshgrid(
            np.arange(0, size, upscale_factor), np.arange(0, size, upscale_factor)
        )
        pixel_locs = np.column_stack((rows.ravel(), cols.ravel()))

        orig_img = np.zeros((8, 8, 3), np.uint8)
        for i, j in pixel_locs:
            orig_img[i // upscale_factor, j // upscale_factor] = image[i, j]

        return orig_img
