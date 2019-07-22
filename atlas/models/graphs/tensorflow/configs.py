from collections.abc import Mapping
from typing import Iterator, Any


class Parameters(Mapping[str, Any]):
    def __init__(self):
        self.__dict__['mapping'] = {}

    def __getitem__(self, key: str) -> Any:
        return self.mapping.__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return self.mapping.__setitem__(key, value)

    def __getattr__(self, item: str):
        return self.mapping.__getitem__(item)

    def __setattr__(self, key: str, value: Any):
        return self.mapping.__setitem__(key, value)

    def get(self, key: str, default: Any):
        if key in self.mapping:
            return self.mapping[key]

        return default

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[str]:
        return self.mapping.__iter__()


class HyperParameters(Parameters):
    """
    Instances of this class are meant to hold high-level hyper-parameters of the network,
    such as batch_size, node dimensions, number of hidden layers, RNN cell types etc.
    These are intended to be largely independent of the training data
    """
    pass


class DataParameters(Parameters):
    """
    These parameters are intended to be more a function of the training-data rather than being a hyper-parameter.
    For example, number of edge types, number of class labels.

    Although in principle, HyperParameters and DataParameters will never be distinguished internally
    during model definition, a clear separation helps in writing the preprocessors which are meant to extract
    these data parameters out of training data sets.

    There are no restrictions on moving any item from DataParameters to Hyper-parameters or vice-versa.
    """
    pass
