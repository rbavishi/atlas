import collections
from abc import abstractmethod, ABC
from typing import Callable, Tuple, Dict, Any, Optional, Set

from atlas.models.core import GeneratorModel


def operator(func):
    setattr(func, "_is_generator_op", True)
    return func


def is_operator(func):
    return getattr(func, "_is_generator_op", False)


class Strategy(ABC):
    def __init__(self):
        self.known_ops: Set[str] = {k for k in dir(self) if is_operator(getattr(self, k))}

    def init(self):
        pass

    def finish(self):
        pass

    def init_run(self):
        pass

    def finish_run(self):
        pass

    @abstractmethod
    def is_finished(self):
        pass

    def get_known_ops(self):
        return self.known_ops

    @abstractmethod
    def make_op(self, op_type: str, oid: Optional[str]) -> Callable:
        pass


class IteratorBasedStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()
        self.model: Optional[GeneratorModel] = None

    def set_model(self, model: Optional[GeneratorModel]):
        self.model = model
