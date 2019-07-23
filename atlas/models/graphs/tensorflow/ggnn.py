import tensorflow as tf

from atlas.models.graphs.tensorflow.network import Network
from atlas.models.graphs.tensorflow.configs import Parameters


class GGNN(Network):
    """The original assembly of propagators and output computation as described in
    https://github.com/microsoft/gated-graph-neural-network-samples/blob/master/chem_tensorflow_sparse.py"""
    def __init__(self,
                 params: Parameters,
                 propagator,
                 classifier,
                 optimizer):

        super().__init__(params)
        self.propagator = propagator
        self.classifier = classifier
        self.optimizer = optimizer

    def build(self):
        self.propagator.build()
        self.classifier.build(node_embeddings=self.propagator.ops['final_node_embeddings'])
        self.optimizer.build(loss=self.classifier.ops['loss'])

