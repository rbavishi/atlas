from abc import abstractmethod, ABC
from typing import Dict, List, Optional, Any, Mapping, Iterable, Iterator

from atlas.models.tensorflow import TensorflowModel
from atlas.models.tensorflow.graphs.earlystoppers import EarlyStopper, SimpleEarlyStopper


class Network(TensorflowModel, ABC):
    def __init__(self, params: Mapping[str, Any]):
        super().__init__()
        self.params = params

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

        if len(cur_batch) > 0:
            yield len(cur_batch), self.define_batch(cur_batch, is_training)

    def train(self, training_data: Iterable[Dict], validation_data: Iterable[Dict], num_epochs: int,
              early_stopper: EarlyStopper = None):
        if self.sess is None:
            self.setup()

        if num_epochs < 0:
            if early_stopper is None:
                early_stopper = SimpleEarlyStopper()

        for epoch in range(1, (num_epochs if num_epochs > 0 else 2**32) + 1):
            train_loss = valid_loss = 0
            train_acc = valid_acc = 0
            train_total_graphs = valid_total_graphs = 0

            training_fetch_list = [self.get_op('loss'), self.get_op('accuracy'), self.get_op('train_step')]
            validation_fetch_list = [self.get_op('loss'), self.get_op('accuracy')]

            for num_graphs, batch_data in self.get_batch_iterator(iter(training_data),
                                                                  self.params['batch_size'], is_training=True):
                batch_loss, batch_acc, _ = self.sess.run(training_fetch_list, feed_dict=batch_data)
                train_loss += batch_loss * num_graphs
                train_acc += batch_acc * num_graphs
                train_total_graphs += num_graphs
                print(f"[Training({epoch}/{num_epochs})] "
                      f"Loss: {train_loss/train_total_graphs: .6f} Accuracy: {train_acc / train_total_graphs: .4f}",
                      end='\r')

            print(f"[Training({epoch}/{num_epochs})] "
                  f"Loss: {train_loss / train_total_graphs: .6f} Accuracy: {train_acc / train_total_graphs: .4f}")

            for num_graphs, batch_data in self.get_batch_iterator(iter(validation_data),
                                                                  self.params['batch_size'], is_training=False):
                batch_loss, batch_acc = self.sess.run(validation_fetch_list, feed_dict=batch_data)
                valid_loss += batch_loss * num_graphs
                valid_acc += batch_acc * num_graphs
                valid_total_graphs += num_graphs
                print(f"[Validation({epoch}/{num_epochs})] "
                      f"Loss: {valid_loss / valid_total_graphs: .6f} Accuracy: {valid_acc / valid_total_graphs: .4f}",
                      end='\r')

            print(f"[Validation({epoch}/{num_epochs})] "
                  f"Loss: {valid_loss / valid_total_graphs: .6f} Accuracy: {valid_acc / valid_total_graphs: .4f}")

            if early_stopper is not None:
                if early_stopper.evaluate(valid_acc / valid_total_graphs, valid_loss / valid_total_graphs):
                    break

    def infer(self, data: Iterable[Dict]):
        num_graphs, batch_data = next(self.get_batch_iterator(iter(data), -1, is_training=False))
        return self.sess.run([self.get_op('predictions'), self.get_op('probabilities')], feed_dict=batch_data)

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def define_batch(self, graphs: List[Dict], is_training: bool = True):
        pass

    @abstractmethod
    def get_op(self, name: str):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('sess')
        state.pop('tf_config')
        state.pop('graph')

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class NetworkComponent:
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
