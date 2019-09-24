import os
import pickle
from abc import ABC, abstractmethod
from typing import Any

from atlas.operators import OpInfo


class Saveable(ABC):
    @abstractmethod
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/loader.pkl", 'wb') as f:
            pickle.dump(self.load, f)

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        with open(f"{path}/loader.pkl", 'rb') as f:
            loader = pickle.load(f)

        return loader(path)


class TrainableModel(Saveable, ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, *args, **kwargs):
        pass


class GeneratorModel(TrainableModel, ABC):
    @abstractmethod
    def train(self, data: Any, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, domain: Any, context: Any = None, op_info: OpInfo = None, **kwargs):
        pass

    def init_run(self):
        pass

    def finish_run(self):
        pass
