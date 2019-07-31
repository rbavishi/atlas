from abc import abstractmethod, ABC
from typing import Any, Collection

from atlas import Generator
from atlas.models.encoding import OpEncoder
from atlas.tracing import GeneratorTrace


class OpModel(ABC):
    def __init__(self, encoder: OpEncoder):
        self.encoder = encoder

    @abstractmethod
    def train(self, gen: Generator, data: Any):
        """
        The entry point for training a generator to bias certain execution paths based on the
        input and an end objective. This method intends to cover the class of imitation/supervised
        learning techniques where a model is trained offline on some collected data.

        Args:
            gen (Generator): The generator to train the model for.
            data: The data to train the generator on (usually traces of generator executions)

        """
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
                print(t)
                op_encoder = self.encoder.get_encoder(t.op_name, t.sid, t.oid)
                encoding = op_encoder(t.domain, t.context, choice=t.choice, mode='training')

