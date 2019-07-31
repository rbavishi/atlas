from abc import abstractmethod, ABC
from typing import Any, Collection

from atlas.models.encoding import OpEncoder
from atlas.tracing import GeneratorTrace


class OpModel(ABC):
    def __init__(self, encoder: OpEncoder):
        self.encoder = encoder

    @abstractmethod
    def train(self, gen: 'Generator', data: Any):
        pass


class IndependentOpModel(OpModel):
    """
    A model where each operator makes decisions independently of the other operators in a generator.
    Each operator is backed by a separate model. Therefore, the supplied training data is split up
    into multiple training data-sets for each of the individual operators in the generator.
    """

    def train(self, gen: 'Generator', data: Collection[GeneratorTrace]):
        for d in data:
            for t in d.op_traces:
                op_encoder = self.encoder.get_encoder(t.op_name, t.sid, t.oid)
                encoding = op_encoder(t.domain, t.context)

