import logging

import pandas as pd
import numpy as np
from typing import Callable, Sequence

from atlas import generator
from atlas.stubs import Select, Sequences, Subsets, OrderedSubsets, Product


# ======================================================================= #
# ======================================================================= #
#                        The DataFrame API
# ======================================================================= #
# ======================================================================= #


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
    """DataFrame.as_matrix(self, columns=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])

    choose_default = Select([True, False], fixed_domain=True)
    if choose_default:
        _columns = None
    else:
        _columns = list(OrderedSubsets(_self.columns))

    return _self.as_matrix(columns=_columns), {'self': _self, 'columns': _columns}


@generator(group='pandas', name='df.get_dtype_counts')
def gen_df_get_dtype_counts(inputs, *args, **kwargs):
    """DataFrame.get_dtype_counts(self)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.get_dtype_counts(), {'self': _self}


@generator(group='pandas', name='df.get_ftype_counts')
def gen_df_get_ftype_counts(inputs, *args, **kwargs):
    """DataFrame.get_ftype_counts(self)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.get_ftype_counts(), {'self': _self}


@generator(group='pandas', name='df.select_dtypes')
def gen_df_select_dtypes(inputs, *args, **kwargs):
    """DataFrame.select_dtypes(self, include=None, exclude=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    dtypes = set(map(str, _self.dtypes))
    use_include = Select([True, False], fixed_domain=True)
    if use_include:
        _include = Subsets(dtypes)
        _exclude = None
    else:
        _include = None
        _exclude = Subsets(dtypes)

    return _self.select_dtypes(include=_include, exclude=_exclude), {
        'self': _self, 'include': _include, 'exclude': _exclude
    }

    # ------------------------------------------------------------------- #
    #  Conversion
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.astype')
def gen_df_astype(inputs, output, *args, **kwargs):
    """DataFrame.astype(self, dtype, copy=True, errors='raise')"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    cand_dtypes = {np.dtype('int64'), np.dtype('int32'), np.dtype('float64'), np.dtype('float32'),
                   np.dtype('bool'), int, float, str, bool}

    if isinstance(output, pd.DataFrame):
        cand_dtypes.update(list(output.dtypes))

    _dtype = Select(cand_dtypes)

    return _self.astype(dtype=_dtype, errors='ignore'), {
        'self': _self, 'dtype': _dtype, 'errors': 'ignore'
    }


@generator(group='pandas', name='df.isna')
def gen_df_isna(inputs, *args, **kwargs):
    """DataFrame.isna(self)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])

    return _self.isna(), {
        'self': _self
    }


@generator(group='pandas', name='df.notna')
def gen_df_notna(inputs, *args, **kwargs):
    """DataFrame.notna(self)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])

    return _self.notna(), {
        'self': _self
    }

    # ------------------------------------------------------------------- #
    #  Indexing & Iteration
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.head')
def gen_df_head(inputs, *args, **kwargs):
    """DataFrame.head(self, n=5)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _n = Select(list(range(1, _self.shape[0])))

    return _self.head(n=_n), {
        'self': _self, 'n': _n
    }


@generator(group='pandas', name='df.tail')
def gen_df_tail(inputs, *args, **kwargs):
    """DataFrame.tail(self, n=5)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _n = Select(list(range(1, _self.shape[0])))

    return _self.tail(n=_n), {
        'self': _self, 'n': _n
    }


@generator(group='pandas', name='df.at.__getitem__')
def gen_df_at_getitem(inputs, *args, **kwargs):
    """DataFrame.at.__getitem__(self, key)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _key = (Select(_self.index), Select(_self.columns))

    return _self.at[_key], {
        'self': _self, 'key': _key
    }


@generator(group='pandas', name='df.iat.__getitem__')
def gen_df_iat_getitem(inputs, *args, **kwargs):
    """DataFrame.iat.__getitem__(self, key)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _key = (Select(list(range(_self.shape[0]))),
            Select(list(range(_self.shape[1]))))

    return _self.iat[_key], {
        'self': _self, 'key': _key
    }


@generator(group='pandas', name='df.loc.__getitem__')
def gen_df_loc_getitem(inputs, *args, **kwargs):
    """DataFrame.loc.__getitem__(self, key)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    idx_reversed = Select([True, False], fixed_domain=True)
    col_reversed = Select([True, False], fixed_domain=True)
    idx_start, idx_end = Subsets(list(range(_self.index)), lengths=[2])
    col_start, col_end = Subsets(list(range(_self.columns)), lengths=[2])

    _key = (
        (slice(idx_start, idx_end, 1) if not idx_reversed else slice(idx_end, idx_start, -1)),
        (slice(col_start, col_end, 1) if not col_reversed else slice(col_end, col_start, -1)),
    )

    return _self.iat[_key], {
        'self': _self, 'key': _key
    }


@generator(group='pandas', name='df.iloc.__getitem__')
def gen_df_iloc_getitem(inputs, *args, **kwargs):
    """DataFrame.iloc.__getitem__(self, key)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    idx_reversed = Select([True, False], fixed_domain=True)
    col_reversed = Select([True, False], fixed_domain=True)
    idx_start, idx_end = Subsets(list(range(_self.shape[0])), lengths=[2])
    col_start, col_end = Subsets(list(range(_self.shape[1])), lengths=[2])

    _key = (
        (slice(idx_start, idx_end, 1) if not idx_reversed else slice(idx_end, idx_start, -1)),
        (slice(col_start, col_end, 1) if not col_reversed else slice(col_end, col_start, -1)),
    )

    return _self.iat[_key], {
        'self': _self, 'key': _key
    }


@generator(group='pandas', name='df.lookup')
def gen_df_lookup(inputs, *args, **kwargs):
    """DataFrame.lookup(self, row_labels, col_labels)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _row_labels = list(OrderedSubsets(_self.index,
                                      lengths=list(range(1, min(_self.shape[0], _self.shape[1]) + 1))))
    _col_labels = list(OrderedSubsets(_self.columns, lengths=[len(_row_labels)]))

    return _self.lookup(row_labels=_row_labels, col_labels=_col_labels), {
        'self': _self, 'row_labels': _row_labels, 'col_labels': _col_labels
    }


@generator(group='pandas', name='df.xs')
def gen_df_xs(inputs, *args, **kwargs):
    """DataFrame.xs(self, key, axis=0, level=None, drop_level=True)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _drop_level = Select([True, False], fixed_domain=True)
    _axis = Select([0, 1], fixed_domain=True)

    src = _self.index if _axis == 0 else _self.columns
    if src.nlevels == 1:
        _level = None
        _key = Select(list(src))
    else:
        _level = Subsets(list(range(src.nlevels)))
        level_vals = [src.levels[i] for i in _level]
        _key = list(Product(level_vals))

    return _self.xs(key=_key, axis=_axis, level=_level, drop_level=_drop_level), {
        'self': _self, 'key': _key, 'axis': _axis, 'level': _level, 'drop_level': _drop_level
    }


@generator(group='pandas', name='df.isin')
def gen_df_isin(inputs, *args, **kwargs):
    """DataFrame.isin(self, values)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _values = Select([inp for inp in inputs
                      if isinstance(inp, (list, tuple, pd.Series, dict, pd.DataFrame))])

    return _self.isin(_values), {
        'self': _self, 'values': _values
    }


@generator(group='pandas', name='df.where')
def gen_df_where(inputs, *args, **kwargs):
    """DataFrame.where(self, cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False,
                       raise_on_error=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _cond = Select([inp for inp in inputs if isinstance(inp, (Sequence, pd.DataFrame, Callable))])
    _other = Select([inp for inp in inputs if isinstance(inp, (Sequence, pd.DataFrame, Callable))])

    return _self.where(_cond, other=_other, errors='ignore'), {
        'self': _self, 'cond': _cond, 'other': _other, 'errors': 'ignore'
    }


@generator(group='pandas', name='df.mask')
def gen_df_mask(inputs, *args, **kwargs):
    """DataFrame.mask(self, cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False,
                       raise_on_error=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _cond = Select([inp for inp in inputs if isinstance(inp, (Sequence, pd.DataFrame, Callable))])
    _other = Select([inp for inp in inputs if isinstance(inp, (Sequence, pd.DataFrame, Callable))])

    return _self.mask(_cond, other=_other, errors='ignore'), {
        'self': _self, 'cond': _cond, 'other': _other, 'errors': 'ignore'
    }


@generator(group='pandas', name='df.query')
def gen_df_query(inputs, *args, **kwargs):
    """DataFrame.query(self, expr, inplace=False)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _expr = Select([inp for inp in inputs if isinstance(inp, str)])

    return _self.query(_expr), {
        'self': _self, 'expr': _expr
    }


@generator(group='pandas', name='df.__getitem__')
def gen_df_getitem(inputs, *args, **kwargs):
    """DataFrame.__getitem__(self, key)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    single_col = Select([True, False], fixed_domain=True)
    if single_col:
        _key = Select(_self.columns)
    else:
        _key = list(OrderedSubsets(_self.columns))

    return _self.__getitem__(_key), {
        'self': _self, 'key': _key
    }
