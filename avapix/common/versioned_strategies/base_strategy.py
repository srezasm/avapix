from abc import ABC, abstractmethod
from numpy import ndarray


class BaseStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.TEXT_LENGTH_INDEX = 2
        self.RANDOM_SEED_INDEX = 170
        self.VERSION_NUM_INDEX = 189

    @abstractmethod
    def get_version_num(self) -> int:
        ...

    @abstractmethod
    def __gen_pixel_order__(self, random_seed: int) -> ndarray:
        ...

    @abstractmethod
    def embed(self, text: str, random_seed: int) -> ndarray:
        ...

    @abstractmethod
    def extract(self, image_array: ndarray) -> str:
        ...
