from abc import ABC, abstractmethod
from typing import Any, Optional


class GeneratorModel(ABC):
    @abstractmethod
    def train(self, data: Any, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, domain: Any, context: Any = None, sid: str = ''):
        pass

    # @abstractmethod
    # def save(self, path: str):
    #     pass
    #
    # @staticmethod
    # @abstractmethod
    # def load(self, path: str):
    #     pass
