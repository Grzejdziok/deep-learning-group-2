import torch.nn as nn

from lenet import Lenet300100


class ModelFactory:

    def __init__(self, model_name: str):
        self._model_name = model_name

    def create(self) -> nn.Module:
        if self._model_name == 'lenet300100':
            return Lenet300100()
        else:
            raise NotImplementedError(f"Unknown model name {self._model_name}")
