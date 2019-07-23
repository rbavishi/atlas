from typing import List, Dict, Any, Mapping

import tensorflow as tf

from atlas.models.graphs.tensorflow.classifiers import GGNNGraphClassifier
from atlas.models.graphs.tensorflow.network import Network
from atlas.models.graphs.tensorflow.optimizers import GGNNOptimizer
from atlas.models.graphs.tensorflow.propagators import GGNNPropagator


class GGNN(Network):
    """The original assembly of propagators and output computation as described in
    https://github.com/microsoft/gated-graph-neural-network-samples/blob/master/chem_tensorflow_sparse.py"""

    def __init__(self,
                 params: Mapping[str, Any],
                 propagator=None,
                 classifier=None,
                 optimizer=None):

        super().__init__(params)
        self.propagator = propagator
        self.classifier = classifier
        self.optimizer = optimizer

        #  Setup default components for ease of use
        if self.propagator is None:
            self.propagator = GGNNPropagator(**params)
        if self.classifier is None:
            self.classifier = GGNNGraphClassifier(**params)
        if self.optimizer is None:
            self.optimizer = GGNNOptimizer(**params)

    def build(self):
        self.propagator.build()
        self.classifier.build(node_embeddings=self.propagator.ops['final_node_embeddings'])
        self.optimizer.build(loss=self.classifier.ops['loss'])

    def define_batch(self, graphs: List[Dict], is_training: bool = True):
        batch_dict = {}
        batch_dict.update(self.propagator.define_batch(graphs, is_training) or {})
        batch_dict.update(self.classifier.define_batch(graphs, is_training) or {})
        batch_dict.update(self.optimizer.define_batch(graphs, is_training) or {})

        return batch_dict

