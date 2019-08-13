import pickle

import tensorflow as tf
from abc import ABC, abstractmethod

from atlas.models import TrainableModel


class TensorflowModel(TrainableModel, ABC):
    def __init__(self, random_seed: int = 0):
        self.sess = None
        self.graph = None
        self.tf_config = None

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

    def save(self, path: str):
        super().save(path)
        if self.sess is not None:
            with self.graph.as_default():
                saver = tf.train.Saver()
                saver.save(self.sess, f"{path}/model.weights")

        with open(f"{path}/model", 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(f"{path}/model", 'rb') as f:
            model = pickle.load(f)

        model.setup_graph()
        with model.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(model.sess, f"{path}/model.weights")

        return model
