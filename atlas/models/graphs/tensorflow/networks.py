#  We use relative imports here as the model codebase is intended to be standalone and not tied to Atlas completely
from .propagators import Propagator


class GGNN:
    """The original assembly of propagators and output computation as described in
    https://github.com/microsoft/gated-graph-neural-network-samples/blob/master/chem_tensorflow_sparse.py"""
    def __init__(self,
                 propagator: Propagator):
        self.propagator = propagator
