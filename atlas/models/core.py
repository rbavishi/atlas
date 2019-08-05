from abc import ABC, abstractmethod
from typing import Any, Optional


class OpModel(ABC):
    pass


class GeneratorModel(ABC):
    def get_model(self, sid: str) -> Optional[OpModel]:
        pass

    @abstractmethod
    def train(self, data: Any, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, domain: Any, context: Any = None, sid: str = ''):
        pass
