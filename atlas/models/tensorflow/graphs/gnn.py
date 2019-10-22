from abc import ABC, abstractmethod
from typing import Mapping, Any, Iterator, Dict, Iterable, List, Optional

from atlas.models.tensorflow import TensorflowModel
from atlas.models.tensorflow.graphs.earlystoppers import EarlyStopper


class GNN(TensorflowModel, ABC):
    def __init__(self, params: Mapping[str, Any]):
        super().__init__()
        self.params = params

    def get_batch_number(self, graph_iter: Iterator[Dict], batch_size: int) -> int:
        return (sum([len(g['nodes']) for g in graph_iter]) + batch_size - 1) // batch_size

    def get_batch_iterator(self, graph_iter: Iterator[Dict],
                           batch_size: int, is_training: bool = True) -> Iterator[Dict]:

        node_offset = 0
        cur_batch = []
        for g in graph_iter:
            cur_batch.append(g)
            node_offset += len(g['nodes'])
            if node_offset > batch_size > 0:
                yield len(cur_batch), self.define_batch(cur_batch, is_training)
                node_offset = 0
                cur_batch = []

        if len(cur_batch) > 0:
            yield len(cur_batch), self.define_batch(cur_batch, is_training)

    def train(self, training_data: Iterable[Dict], validation_data: Iterable[Dict], num_epochs: int = 1,
              early_stopper: EarlyStopper = None, **kwargs):
        return super().train(training_data, validation_data,
                             batch_size=self.params['batch_size'], num_epochs=num_epochs, early_stopper=early_stopper)

    def infer(self, data: Iterable[Dict]):
        num_graphs, batch_data = next(self.get_batch_iterator(iter(data), -1, is_training=False))
        return self.sess.run([self.ops['predictions'], self.ops['probabilities']], feed_dict=batch_data)

    def build(self):
        self.build_graph()
        for k, v in self.__dict__.items():
            if isinstance(v, GNNComponent):
                self.ops.update(v.ops)
                self.placeholders.update(v.placeholders)
                self.weights.update(v.weights)

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def define_batch(self, graphs: List[Dict], is_training: bool = True):
        pass


class GNNComponent:
    def __init__(self):
        self.placeholders = {}
        self.weights = {}
        self.ops = {}

    def define_batch(self, graphs: List[Dict], is_training: bool = True) -> Optional[Dict]:
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('placeholders')
        state.pop('weights')
        state.pop('ops')

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.placeholders = {}
        self.weights = {}
        self.ops = {}
