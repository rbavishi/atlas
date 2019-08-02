from abc import abstractmethod, ABC
from typing import Any

from atlas.models.encoding import OpEncoder


class OpModel(ABC):
    def __init__(self, encoder: OpEncoder):
        self.encoder = encoder

    @abstractmethod
    def train(self, gen: 'Generator', data: Any, **kwargs):
        pass


