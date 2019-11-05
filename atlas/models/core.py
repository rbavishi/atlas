from abc import ABC, abstractmethod
from typing import Any

from atlas.operators import OpInfo


class SerializableModel(ABC):
    @abstractmethod
    def serialize(self, path: str):
        pass

    @abstractmethod
    def deserialize(self, path: str):
        pass


class TrainableModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, *args, **kwargs):
        pass


class TrainableSerializableModel(SerializableModel, TrainableModel, ABC):
    pass


class GeneratorModel(ABC):
    @abstractmethod
    def infer(self, domain: Any, context: Any = None, op_info: OpInfo = None, **kwargs):
        pass
