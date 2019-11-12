import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Iterator, Iterable, Dict, List, Optional

import tensorflow as tf
from atlas.models.core import TrainableModel, SerializableModel
from atlas.models.tensorflow.graphs.earlystoppers import EarlyStopper, SimpleEarlyStopper


class TensorflowModel(TrainableModel, SerializableModel, ABC):
    def __init__(self, random_seed: int = 0):
        self.sess = None
        self.graph = None
        self.tf_config = None

        self.placeholders = {}
        self.weights = {}
        self.ops = {}

        self.random_seed = random_seed

    def set_random_seed(self, seed: int):
        self.random_seed = seed

    def setup_graph(self):
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=self.tf_config)

        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.build()

    def initialize_variables(self):
        with self.graph.as_default():
            self.sess.run(tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer()))

    def setup(self):
        self.setup_graph()
        self.initialize_variables()

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def get_batch_number(self, graph_iter: Iterator[Dict], batch_size: int) -> int:
        pass

    @abstractmethod
    def get_batch_iterator(self, data_iter: Iterator, batch_size: int, is_training: bool = True) -> Iterator:
        pass

    def train(self,
              training_data: Iterable, validation_data: Iterable,
              batch_size: int = 128,
              num_epochs: int = 1,
              early_stopper: EarlyStopper = None,
              **kwargs):

        if self.sess is None:
            self.setup()

        if num_epochs < 0:
            if early_stopper is None:
                early_stopper = SimpleEarlyStopper()

        history: List[Dict] = []

        #  Machinery to save the best performing models
        best_acc = 0
        best_loss = 10**10
        saved = False
        with self.graph.as_default():
            saver = tf.train.Saver()

        num_batches_train = self.get_batch_number(iter(training_data), batch_size)
        num_batches_valid = self.get_batch_number(iter(validation_data), batch_size)

        tmpdir: Optional[str] = None
        try:
            tmpdir = tempfile.mkdtemp()
            for epoch in range(1, (num_epochs if num_epochs >= 0 else 2**32) + 1):
                history.append({'epoch': epoch})

                train_loss = valid_loss = 0
                train_acc = valid_acc = 0
                num_datapoints = 0

                training_fetch_list = [self.ops['loss'], self.ops['accuracy'], self.ops['train_step']]
                validation_fetch_list = [self.ops['loss'], self.ops['accuracy']]

                for i, (num_datapoints_batch, batch_data) in enumerate(self.get_batch_iterator(
                                        iter(training_data), batch_size, is_training=True)):
                    batch_loss, batch_acc, _ = self.sess.run(training_fetch_list, feed_dict=batch_data)
                    train_loss += batch_loss * num_datapoints_batch
                    train_acc += batch_acc * num_datapoints_batch
                    num_datapoints += num_datapoints_batch
                    print(f"[Training] Epoch: {epoch}/{num_epochs}\tBatch: {i}/{num_batches_train}\t"
                          f"Loss: {train_loss / num_datapoints: .6f}\tAccuracy: {train_acc / num_datapoints: .4f}",
                          end='\r')

                print(f"[Training] Epoch: {epoch}/{num_epochs}\tBatch: {num_batches_train}\t"
                      f"Loss: {train_loss / num_datapoints: .6f}\tAccuracy: {train_acc / num_datapoints: .4f}")

                history[-1].update({
                    'train_loss': train_loss / num_datapoints,
                    'train_acc': train_acc / num_datapoints,
                })

                num_datapoints = 0
                for i, (num_datapoints_batch, batch_data) in enumerate(self.get_batch_iterator(
                                        iter(validation_data), batch_size, is_training=False)):
                    batch_loss, batch_acc = self.sess.run(validation_fetch_list, feed_dict=batch_data)
                    valid_loss += batch_loss * num_datapoints_batch
                    valid_acc += batch_acc * num_datapoints_batch
                    num_datapoints += num_datapoints_batch
                    print(f"[Validation] Epoch: {epoch}/{num_epochs}\tBatch: {i}/{num_batches_valid}\t"
                          f"Loss: {valid_loss / num_datapoints: .6f}\tAccuracy: {valid_acc / num_datapoints: .4f}",
                          end='\r')

                print(f"[Validation] Epoch: {epoch}/{num_epochs}\tBatch: {num_batches_valid}\t"
                      f"Loss: {valid_loss / num_datapoints: .6f} Accuracy: {valid_acc / num_datapoints: .4f}")

                history[-1].update({
                    'valid_loss': valid_loss / num_datapoints,
                    'valid_acc': valid_acc / num_datapoints,
                })

                cur_acc = valid_acc / num_datapoints
                cur_loss = valid_loss / num_datapoints

                if (cur_acc > best_acc) or (cur_acc == best_acc and cur_loss < best_loss):
                    saver.save(self.sess, f"{tmpdir}/model.weights",
                               write_meta_graph=False, write_state=False)
                    best_acc = cur_acc
                    best_loss = cur_loss
                    saved = True

                if early_stopper is not None:
                    if early_stopper.evaluate(valid_acc / num_datapoints, valid_loss / num_datapoints):
                        break

            if saved:
                saver.restore(self.sess, f"{tmpdir}/model.weights")

            return history

        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir)

    @abstractmethod
    def infer(self, data: Iterator):
        pass

    def warmup(self):
        #  Useful for speeding up the first inference
        pass

    def serialize(self, path: str):
        if self.sess is not None:
            with self.graph.as_default():
                saver = tf.train.Saver()
                saver.save(self.sess, f"{path}/model.weights")

    def deserialize(self, path: str):
        self.setup_graph()
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, f"{path}/model.weights")
            self.warmup()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('sess')
        state.pop('tf_config')
        state.pop('graph')
        state.pop('placeholders')
        state.pop('weights')
        state.pop('ops')

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.placeholders = {}
        self.weights = {}
        self.ops = {}
