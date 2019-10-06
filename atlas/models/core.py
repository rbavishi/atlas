import os
import pickle
from abc import ABC, abstractmethod
from typing import Any

from atlas.operators import OpInfo


class AtlasModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, *args, **kwargs):
        pass

    @abstractmethod
    def serialize(self, path: str):
        pass

    @abstractmethod
    def deserialize(self, path: str):
        pass


class GeneratorModel(AtlasModel, ABC):
    @abstractmethod
    def train(self, data: Any, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, domain: Any, context: Any = None, op_info: OpInfo = None, **kwargs):
        pass
