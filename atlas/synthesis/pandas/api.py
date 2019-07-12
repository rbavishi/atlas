import logging

import pandas as pd

from atlas import generator
from atlas.stubs import Select, Sequences, Subsets, OrderedSubsets


# ----------------------------------------------------------------------- #
#  Attributes
# ----------------------------------------------------------------------- #
#  These are not callable, but its values are sometimes useful, and hence
#  they are included in synthesis
# ----------------------------------------------------------------------- #


@generator(group='pandas', name='df.index')
def gen_df_index(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.index, {'self': _self}


@generator(group='pandas', name='df.columns')
def gen_df_columns(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.columns, {'self': _self}


@generator(group='pandas', name='df.dtypes')
def gen_df_dtypes(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.dtypes, {'self': _self}


@generator(group='pandas', name='df.ftypes')
def gen_df_ftypes(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.ftypes, {'self': _self}


@generator(group='pandas', name='df.values')
def gen_df_values(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.values, {'self': _self}


@generator(group='pandas', name='df.axes')
def gen_df_axes(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.axes, {'self': _self}


@generator(group='pandas', name='df.ndim')
def gen_df_ndim(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.ndim, {'self': _self}


@generator(group='pandas', name='df.size')
def gen_df_size(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.size, {'self': _self}


@generator(group='pandas', name='df.shape')
def gen_df_shape(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.shape, {'self': _self}


@generator(group='pandas', name='df.T')
def gen_df_T(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.T, {'self': _self}


# ----------------------------------------------------------------------- #
#  Methods
# ----------------------------------------------------------------------- #

# ------------------------------------------------------------------- #
#  Attributes & Underlying Data
# ------------------------------------------------------------------- #


@generator(group='pandas', name='df.as_matrix')
def gen_df_as_matrix(inputs, *args, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])

    choose_default = Select([True, False], fixed_domain=True)
    if choose_default:
        _columns = None
    else:
        _columns = list(OrderedSubsets(_self.columns))

    return _self.as_matrix(columns=_columns), {'self': _self, 'columns': _columns}

