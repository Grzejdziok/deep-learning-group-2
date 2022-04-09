import torch.nn as nn

from lenet import Lenet300100
from conv2 import Conv2
from conv4 import Conv4
from conv6 import Conv6


class ModelFactory:

    MODELS = {
        'lenet300100': Lenet300100,
        'conv2': Conv2,
        'conv4': Conv4,
        'conv6': Conv6,
    }

    def __init__(self, model_name: str):
        self._model_name = model_name
        assert self._model_name in self.MODELS.keys()

    def create(self) -> nn.Module:
        return self.MODELS[self._model_name]()
