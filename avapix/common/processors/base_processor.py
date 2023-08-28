from abc import ABC, abstractmethod
from numpy import ndarray

TEXT_LENGTH_INDEX = 2
RANDOM_SEED_INDEX = 170
VERSION_NUM_INDEX = 189

class BaseProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.TEXT_LENGTH_INDEX = TEXT_LENGTH_INDEX
        self.RANDOM_SEED_INDEX = RANDOM_SEED_INDEX
        self.VERSION_NUM_INDEX = VERSION_NUM_INDEX

    @abstractmethod
    def get_version_num(self) -> int:
        ...

    @abstractmethod
    def embed(self, text: str, random_seed: int) -> ndarray:
        ...

    @abstractmethod
    def extract(self, image_array: ndarray) -> str:
        ...

    @abstractmethod
    def __gen_pixel_order__(self, random_seed: int) -> ndarray:
        ...
