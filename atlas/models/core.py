import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional


class GeneratorModel(ABC):
    @abstractmethod
    def train(self, data: Any, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, domain: Any, context: Any = None, sid: str = ''):
        pass

    @abstractmethod
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/loader.pkl", 'wb') as f:
            pickle.dump(self.load, f)

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'GeneratorModel':
        with open(f"{path}/loader.pkl", 'rb') as f:
            loader = pickle.load(f)

        return loader(path)
