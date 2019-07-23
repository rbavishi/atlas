from abc import abstractmethod, ABC

import tensorflow as tf
from typing import Dict

from atlas.models.graphs.tensorflow.configs import Parameters


class Network(ABC):
    def __init__(self, params: Parameters):
        self.sess = None
        self.graph = None
        self.tf_config = None

        self.params = params

    def setup_session(self):
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=self.tf_config)

        with self.graph.as_default():
            tf.set_random_seed(self.params.get('random_seed', 0))
            self.build()
            self.sess.run(tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer()))

    @abstractmethod
    def build(self):
        pass


class NetworkComponent:
    def __init__(self):
        self.placeholders = {}
        self.weights = {}
        self.ops = {}

    def define_batch(self, graphs, is_training: bool = True) -> Dict:
        return {}
