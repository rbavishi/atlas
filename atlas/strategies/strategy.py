from abc import abstractmethod, ABC
from typing import Callable, Optional, Set

from atlas.models.core import GeneratorModel
from atlas.operators import DefaultOpMethodResolver, OpInfo


def operator(func):
    setattr(func, "_is_generator_op", True)
    return func


def method(func):
    setattr(func, "_is_generator_method", True)
    return func


def is_operator(func):
    return getattr(func, "_is_generator_op", False)


def is_method(func):
    return getattr(func, "_is_generator_method", False)


class Strategy(ABC, DefaultOpMethodResolver):
    def __init__(self):
        self.known_ops: Set[str] = {k for k in dir(self) if is_operator(getattr(self, k))}
        self.known_methods: Set[str] = {k for k in dir(self) if is_method(getattr(self, k))}

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

    def get_known_methods(self):
        return self.known_methods

    def generic_call(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     **kwargs):
        pass


class IteratorBasedStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()
        self.model: Optional[GeneratorModel] = None

    def set_model(self, model: Optional[GeneratorModel]):
        self.model = model
