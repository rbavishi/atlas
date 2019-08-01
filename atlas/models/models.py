from abc import abstractmethod, ABC
from typing import Any, Collection

from atlas.models.encoding import OpEncoder
from atlas.tracing import GeneratorTrace


class OpModel(ABC):
    def __init__(self, encoder: OpEncoder):
        self.encoder = encoder

    @abstractmethod
    def train(self, gen: 'Generator', data: Any, **kwargs):
        pass


class TraceImitationOpModel(OpModel, ABC):
    @abstractmethod
    def train(self, gen: 'Generator', traces: Collection[GeneratorTrace], **kwargs):
        pass

