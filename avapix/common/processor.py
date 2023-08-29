from typing import Literal
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray

from avapix.common.processors.base_processor import VERSION_NUM_INDEX
from avapix.common.processors.v1_0 import ProcessorV1_0
from avapix.common.processors.v1_1 import ProcessorV1_1
from avapix.common.models.avapix_models import V1
from avapix.common.utils import *


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
        version: Literal["v1.0", "v1.1"] = "v1.1",
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
            gen_result = self.model.gen(to_tensor(raw_embedded_img, self.device))

        gen_result = to_numpy(gen_result)

        # Replace the originally embedded color channels
        mask = raw_embedded_img != 0
        gen_result[mask] = raw_embedded_img[mask]

        return gen_result

    def extract(self, image_array: ndarray):
        processor = None

        version = image_array.reshape(-1)[VERSION_NUM_INDEX]

        if version == 100:
            processor = ProcessorV1_0()
        elif version == 110:
            processor = ProcessorV1_1()
        else:
            raise ValueError(f"Invalid version: {version}")

        return processor.extract(image_array)
