import logging
import typing

import pandas as pd
import numpy as np
from typing import Callable

from atlas import generator
from atlas.stubs import Select, Sequence, Subset, OrderedSubset, Product
from atlas.synthesis.numpy.utils import get_non_1_prime_factors


@generator(group='numpy', name='ndarray.flatten')
def gen_ndarray_flatten(inputs, *args, **kwargs):
    """ndarray.flatten(self, order='C')"""
    _self = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])
    _order = Select(['C', 'F', 'A', 'K'])
    return _self.flatten(order=_order), {
        'self': _self, 'order': _order
    }


@generator(group='numpy', name='ndarray.transpose')
def gen_ndarray_transpose(inputs, *args, **kwargs):
    """ndarray.transpose(self, *axes)"""

    _self = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    _axes = None
    use_axis = Select([True, False])
    if use_axis:
        axis_indices = range(len(_self.shape))
        _axes = OrderedSubset(axis_indices, lengths=[len(axis_indices)])

    return _self.transpose(_axes), {
        'self': _self, 'axes': _axes
    }


@generator(group='numpy', name='ndarray.reshape')
def gen_ndarray_reshape(inputs, *args, **kwargs):
    """ndarray.reshape(self, shape, order='C')"""

    _self = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])
    # How many dimensions does one want to reshape to?
    n_dims = Select(range(1, 6))
    if n_dims == 1:
        _shape = _self.size
    else:
        non_1_factors = get_non_1_prime_factors(_self.size)

        _shape = []
        for i in range(n_dims - 1):
            factors_to_use = Subset(non_1_factors, include_empty=True)
            dimension_size = 1
            for f in factors_to_use:
                dimension_size *= f
                non_1_factors.remove(f)
            _shape.append(dimension_size)
        # Make sure in the end we get all the factors
        dimension_size = 1
        for f in non_1_factors:
            dimension_size *= f
        _shape.append(dimension_size)

    _order = Select(['C', 'F', 'A'])

    return _self.reshape(_shape, order=_order), {
        'self': _self, 'shape': _shape, 'order': _order
    }
