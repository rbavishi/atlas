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
    idx_start, idx_end = Subsets(list(_self.index), lengths=[2])
    col_start, col_end = Subsets(list(_self.columns), lengths=[2])

    _key = (
        (slice(idx_start, idx_end, 1) if not idx_reversed else slice(idx_end, idx_start, -1)),
        (slice(col_start, col_end, 1) if not col_reversed else slice(col_end, col_start, -1)),
    )

    return _self.loc[_key], {
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

    return _self.iloc[_key], {
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

    def is_valid_cond(cond):
        if isinstance(cond, pd.DataFrame) and \
                len(cond.select_dtypes(include=[bool, np.dtype('bool')]).columns) != len(cond.columns):
            return False

        return True

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _cond = Select([inp for inp in inputs
                    if isinstance(inp, (Sequence, pd.DataFrame, Callable)) and is_valid_cond(inp)])
    _other = Select([inp for inp in inputs if isinstance(inp, (Sequence, pd.DataFrame, Callable))])

    return _self.where(_cond, other=_other, errors='ignore'), {
        'self': _self, 'cond': _cond, 'other': _other, 'errors': 'ignore'
    }


@generator(group='pandas', name='df.mask')
def gen_df_mask(inputs, *args, **kwargs):
    """DataFrame.mask(self, cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False,
                       raise_on_error=None)"""

    def is_valid_cond(cond):
        if isinstance(cond, pd.DataFrame) and \
                len(cond.select_dtypes(include=[bool, np.dtype('bool')]).columns) != len(cond.columns):
            return False

        return True

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _cond = Select([inp for inp in inputs
                    if isinstance(inp, (Sequence, pd.DataFrame, Callable)) and is_valid_cond(inp)])
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

    # ------------------------------------------------------------------- #
    #  Binary Operator Functions
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.add')
def gen_df_add(inputs, *args, **kwargs):
    """DataFrame.add(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.add(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.sub')
def gen_df_sub(inputs, *args, **kwargs):
    """DataFrame.sub(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.sub(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.mul')
def gen_df_mul(inputs, *args, **kwargs):
    """DataFrame.mul(self, other, axis='columns', level=None, fill_value=None)"""

    #  Only return something if all the columns
    #  are of integer types. Otherwise things can
    #  get nasty and cause memory issues
    #  For example 1000 * 1000 * 1000 * "abcd"
    #  would wreak havoc on the system
    #  TODO : Is there a better way without restricting functionality?
    def validate_self(val):
        if len(val.select_dtypes(include=np.number).columns) != len(val.columns):
            return False

        return True

    def validate_other(val):
        if isinstance(val, pd.DataFrame):
            if len(val.select_dtypes(include=np.number).columns) != len(val.columns):
                return False

        elif isinstance(val, pd.Series):
            if not issubclass(val.dtype.type, np.number):
                return False

        elif isinstance(val, str):
            return False

        return True

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame) and validate_self(inp)])
    _other = Select([inp for inp in inputs if validate_other(inp)])

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.mul(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.div')
def gen_df_div(inputs, *args, **kwargs):
    """DataFrame.div(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.div(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.truediv')
def gen_df_truediv(inputs, *args, **kwargs):
    """DataFrame.truediv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.truediv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.floordiv')
def gen_df_floordiv(inputs, *args, **kwargs):
    """DataFrame.floordiv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.floordiv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.mod')
def gen_df_mod(inputs, *args, **kwargs):
    """DataFrame.mod(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.mod(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.pow')
def gen_df_pow(inputs, *args, **kwargs):
    """DataFrame.pow(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.pow(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.radd')
def gen_df_radd(inputs, *args, **kwargs):
    """DataFrame.radd(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.radd(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rsub')
def gen_df_rsub(inputs, *args, **kwargs):
    """DataFrame.rsub(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.rsub(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rmul')
def gen_df_rmul(inputs, *args, **kwargs):
    """DataFrame.rmul(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.rmul(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rdiv')
def gen_df_rdiv(inputs, *args, **kwargs):
    """DataFrame.rdiv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.rdiv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rtruediv')
def gen_df_rtruediv(inputs, *args, **kwargs):
    """DataFrame.rtruediv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.rtruediv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rfloordiv')
def gen_df_rfloordiv(inputs, *args, **kwargs):
    """DataFrame.rfloordiv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.rfloordiv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rmod')
def gen_df_rmod(inputs, *args, **kwargs):
    """DataFrame.rmod(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.rmod(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rpow')
def gen_df_rpow(inputs, *args, **kwargs):
    """DataFrame.rpow(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    fill_value_cands = [inp for inp in inputs if isinstance(inp, (int, float))]
    if len(fill_value_cands) > 0:
        _fill_value = Select([None] + fill_value_cands)
    else:
        _fill_value = None

    return _self.rpow(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.lt')
def gen_df_lt(inputs, *args, **kwargs):
    """DataFrame.lt(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    return _self.lt(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.gt')
def gen_df_gt(inputs, *args, **kwargs):
    """DataFrame.gt(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    return _self.gt(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.le')
def gen_df_le(inputs, *args, **kwargs):
    """DataFrame.le(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    return _self.le(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.ge')
def gen_df_ge(inputs, *args, **kwargs):
    """DataFrame.ge(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    return _self.ge(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.ne')
def gen_df_ne(inputs, *args, **kwargs):
    """DataFrame.ne(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    return _self.ne(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.eq')
def gen_df_eq(inputs, *args, **kwargs):
    """DataFrame.eq(self, other, axis='columns', level=None, fill_value=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select(inputs)

    if isinstance(_other, pd.Series):
        _axis = Select(['columns', 'index'], fixed_domain=True)
    else:
        _axis = 'columns'

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    return _self.eq(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.combine')
def gen_df_combine(inputs, *args, **kwargs):
    """DataFrame.combine(self, other, func, fill_value=None, overwrite=True)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _func = Select([inp for inp in inputs if isinstance(inp, Callable)])
    _overwrite = Select([True, False], fixed_domain=True)

    fill_val_cands = [inp for inp in inputs if np.isscalar(inp)]
    if len(fill_val_cands) > 0:
        _fill_value = Select([None] + fill_val_cands)
    else:
        _fill_value = None

    return _self.combine(other=_other, func=_func, fill_value=_fill_value, overwrite=_overwrite), {
        'self': _self, 'other': _other, 'func': _func, 'fill_value': _fill_value, 'overwrite': _overwrite
    }


@generator(group='pandas', name='df.combine_first')
def gen_df_combine_first(inputs, *args, **kwargs):
    """DataFrame.combine_first(self, other)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])

    return _self.combine_first(other=_other), {
        'self': _self, 'other': _other
    }
