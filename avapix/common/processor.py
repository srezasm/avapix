from typing import Literal
from numpy import ndarray

from avapix.common.processors.base_processor import VERSION_NUM_INDEX
from avapix.common.processors.v1_0 import ProcessorV1_0
from avapix.common.processors.v1_1 import ProcessorV1_1

class Processor:
    _VERSIONS = Literal['v1.0', 'v1.1']
    def embed(self, text: str, random_seed: int = 42, version: _VERSIONS = 'v1.1'):
        processor = None

        if version == 'v1.0':
            processor = ProcessorV1_0()
        elif version == 'v1.1':
            processor = ProcessorV1_1()
        else:
            raise ValueError(f'Invalid processor version: {version}')
        
        return processor.embed(text, random_seed)

    def extract(self, image_array: ndarray):
        processor = None

        version = int(
            image_array.reshape(-1)[VERSION_NUM_INDEX])

        if version == 100:
            processor = ProcessorV1_0()
        elif version == 110:
            processor = ProcessorV1_1()
        else:
            raise ValueError(f'Invalid version: {version}')
        
        return processor.extract(image_array)
