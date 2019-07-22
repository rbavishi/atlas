import tensorflow as tf

from atlas.models.graphs.tensorflow import Network
from atlas.models.graphs.tensorflow.configs import HyperParameters, DataParameters


class GGNN(Network):
    """The original assembly of propagators and output computation as described in
    https://github.com/microsoft/gated-graph-neural-network-samples/blob/master/chem_tensorflow_sparse.py"""
    def __init__(self,
                 propagator,
                 classifier,
                 optimizer,
                 hyper_params: HyperParameters,
                 data_params: DataParameters = None):

        super().__init__()
        self.propagator = propagator
        self.classifier = classifier
        self.optimizer = optimizer

        self.hyper_params = hyper_params
        self.data_params = data_params if data_params is not None else DataParameters()

    def build(self):
        params = {**self.hyper_params, **self.data_params}

        self.propagator.build(**params)
        self.classifier.build(self.propagator.ops['final_node_embeddings'], **params)
        self.optimizer.build(self.classifier.ops['loss'], **params)

