import pandas as pd

from atlas import generator
from atlas.stubs import Select, Sequences, Subsets, OrderedSubsets


@generator(group='pandas', name='df.index')
def gen_df_index(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].index, args


@generator(group='pandas', name='df.columns')
def gen_df_columns(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].columns, args


@generator(group='pandas', name='df.dtypes')
def gen_df_dtypes(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].dtypes, args


@generator(group='pandas', name='df.ftypes')
def gen_df_ftypes(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].ftypes, args


@generator(group='pandas', name='df.values')
def gen_df_values(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].values, args


@generator(group='pandas', name='df.axes')
def gen_df_axes(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].axes, args


@generator(group='pandas', name='df.ndim')
def gen_df_ndim(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].ndim, args


@generator(group='pandas', name='df.size')
def gen_df_size(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].size, args


@generator(group='pandas', name='df.shape')
def gen_df_shape(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].shape, args


@generator(group='pandas', name='df.T')
def gen_df_T(inputs, *args, **kwargs):
    args = {}
    args['self'] = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return args['self'].T, args
