import logging

import pandas as pd
import numpy as np
from typing import Callable, Sequence

from atlas import generator
from atlas.stubs import Select, Sequence, Subset, OrderedSubset, Product


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
        _columns = list(OrderedSubset(_self.columns))

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
        _include = Subset(dtypes)
        _exclude = None
    else:
        _include = None
        _exclude = Subset(dtypes)

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
    idx_start, idx_end = Subset(list(_self.index), lengths=[2])
    col_start, col_end = Subset(list(_self.columns), lengths=[2])

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
    idx_start, idx_end = Subset(list(range(_self.shape[0])), lengths=[2])
    col_start, col_end = Subset(list(range(_self.shape[1])), lengths=[2])

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
    _row_labels = list(OrderedSubset(_self.index,
                                      lengths=list(range(1, min(_self.shape[0], _self.shape[1]) + 1))))
    _col_labels = list(OrderedSubset(_self.columns, lengths=[len(_row_labels)]))

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
        _level = Subset(list(range(src.nlevels)))
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
        _key = list(OrderedSubset(_self.columns))

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

    # ------------------------------------------------------------------- #
    #  Function application, GroupBy & Window
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.apply')
def gen_df_apply(inputs, *args, **kwargs):
    """DataFrame.apply(self, func, axis=0, broadcast=False, raw=False, reduce=None)"""


@generator(group='pandas', name='df.applymap')
def gen_df_applymap(inputs, *args, **kwargs):
    """DataFrame.applymap(self, func)"""


@generator(group='pandas', name='df.agg')
def gen_df_agg(inputs, *args, **kwargs):
    """DataFrame.agg(self, func, axis=0)"""


@generator(group='pandas', name='df.transform')
def gen_df_transform(inputs, *args, **kwargs):
    """DataFrame.transform(self, func)"""


@generator(group='pandas', name='df.groupby')
def gen_df_groupby(inputs, *args, **kwargs):
    """DataFrame.groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False)"""

    # ------------------------------------------------------------------- #
    #  Computations/Descriptive Stats
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.abs')
def gen_df_abs(inputs, *args, **kwargs):
    """DataFrame.abs(self)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    return _self.abs(), {
        'self': _self
    }


@generator(group='pandas', name='df.all')
def gen_df_all(inputs, *args, **kwargs):
    """DataFrame.all(self, axis=None, bool_only=None, skipna=None, level=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _bool_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.all(axis=_axis, bool_only=_bool_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'bool_only': _bool_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.any')
def gen_df_any(inputs, *args, **kwargs):
    """DataFrame.any(self, axis=None, bool_only=None, skipna=None, level=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _bool_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.any(axis=_axis, bool_only=_bool_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'bool_only': _bool_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.clip')
def gen_df_clip(inputs, output, *args, **kwargs):
    """DataFrame.clip(self, lower=None, upper=None, axis=None, inplace=False)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    lower_cands = [inp for inp in inputs if isinstance(inp, (float, np.floating, int, np.number))]
    upper_cands = [inp for inp in inputs if isinstance(inp, (float, np.floating, int, np.number))]

    if isinstance(output, pd.DataFrame):
        lower_cands.append(np.min(output.select_dtypes(include=np.number).values))
        upper_cands.append(np.max(output.select_dtypes(include=np.number).values))

    _lower = Select(lower_cands)
    _upper = Select(upper_cands)

    return _self.clip(lower=_lower, upper=_upper), {
        'self': _self, 'lower': _lower, 'upper': _upper
    }


@generator(group='pandas', name='df.clip_lower')
def gen_df_clip_lower(inputs, output, *args, **kwargs):
    """DataFrame.clip_lower(self, threshold, axis=None, inplace=False)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    threshold_cands = [inp for inp in inputs if isinstance(inp, (float, np.floating, int, np.number))]
    if isinstance(output, pd.DataFrame):
        threshold_cands.append(np.min(output.select_dtypes(include=np.number).values))

    _threshold = Select(threshold_cands)
    return _self.clip_lower(threshold=_threshold), {
        'self': _self, 'threshold': _threshold
    }


@generator(group='pandas', name='df.clip_upper')
def gen_df_clip_upper(inputs, output, *args, **kwargs):
    """DataFrame.clip_upper(self, threshold, axis=None, inplace=False)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    threshold_cands = [inp for inp in inputs if isinstance(inp, (float, np.floating, int, np.number))]
    if isinstance(output, pd.DataFrame):
        threshold_cands.append(np.max(output.select_dtypes(include=np.number).values))

    _threshold = Select(threshold_cands)
    return _self.clip_upper(threshold=_threshold), {
        'self': _self, 'threshold': _threshold
    }


@generator(group='pandas', name='df.corr')
def gen_df_corr(inputs, *args, **kwargs):
    """DataFrame.corr(self, method='pearson', min_periods=1)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _min_periods = Select([1] + [inp for inp in inputs if isinstance(inp, (int, np.number))])
    _method = Select(['pearson', 'kendall', 'spearman'], fixed_domain=True)

    return _self.corr(min_periods=_min_periods, method=_method), {
        'self': _self, 'min_periods': _min_periods, 'method': _method
    }


@generator(group='pandas', name='df.corrwith')
def gen_df_corrwith(inputs, *args, **kwargs):
    """DataFrame.corrwith(self, other, axis=0, drop=False)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _drop = Select([False, True], fixed_domain=True)
    _axis = Select([0, 1], fixed_domain=True)

    return _self.corrwith(_other, axis=_axis, drop=_drop), {
        'self': _self, 'other': _other, 'axis': _axis, 'drop': _drop
    }


@generator(group='pandas', name='df.count')
def gen_df_count(inputs, *args, **kwargs):
    """DataFrame.count(self, axis=0, level=None, numeric_only=False)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([False, True], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    return _self.count(axis=_axis, level=_level, numeric_only=_numeric_only), {
        'self': _self, 'axis': _axis, 'level': _level, 'numeric_only': _numeric_only
    }


@generator(group='pandas', name='df.cov')
def gen_df_cov(inputs, *args, **kwargs):
    """DataFrame.cov(self, min_periods=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _min_periods = Select([None] + [inp for inp in inputs if isinstance(inp, (int, np.number))])

    return _self.cov(min_periods=_min_periods), {
        'self': _self, 'min_periods': _min_periods
    }


@generator(group='pandas', name='df.cummax')
def gen_df_cummax(inputs, *args, **kwargs):
    """DataFrame.cummax(self, axis=None, skipna=True)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    return _self.cummax(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.cummin')
def gen_df_cummin(inputs, *args, **kwargs):
    """DataFrame.cummin(self, axis=None, skipna=True)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    return _self.cummin(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.cumprod')
def gen_df_cumprod(inputs, *args, **kwargs):
    """DataFrame.cumprod(self, axis=None, skipna=True)"""

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

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame) and validate_self(inp)])
    _axis = Select([0, 1], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    return _self.cumprod(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.cumsum')
def gen_df_cumsum(inputs, *args, **kwargs):
    """DataFrame.cumsum(self, axis=None, skipna=True)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    return _self.cumsum(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.diff')
def gen_df_diff(inputs, *args, **kwargs):
    """DataFrame.diff(self, periods=1, axis=0)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _periods = Select([1] + [inp for inp in inputs if isinstance(inp, (int, np.number))])
    _axis = Select([0, 1], fixed_domain=True)

    return _self.diff(axis=_axis, periods=_periods), {
        'self': _self, 'axis': _axis, 'periods': _periods
    }


@generator(group='pandas', name='df.eval')
def gen_df_eval(inputs, *args, **kwargs):
    """DataFrame.eval(self, expr, inplace=False)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _expr = Select([inp for inp in inputs if isinstance(inp, str)])

    return _self.eval(_expr), {
        'self': _self, 'expr': _expr
    }


@generator(group='pandas', name='df.kurt')
def gen_df_kurt(inputs, *args, **kwargs):
    """DataFrame.kurt(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.kurt(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.mad')
def gen_df_mad(inputs, *args, **kwargs):
    """DataFrame.mad(self, axis=None, skipna=None, level=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.mad(axis=_axis, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.max')
def gen_df_max(inputs, *args, **kwargs):
    """DataFrame.max(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.max(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.mean')
def gen_df_mean(inputs, *args, **kwargs):
    """DataFrame.max(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.mean(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.median')
def gen_df_median(inputs, *args, **kwargs):
    """DataFrame.median(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.median(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.min')
def gen_df_min(inputs, *args, **kwargs):
    """DataFrame.min(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.min(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.mode')
def gen_df_mode(inputs, *args, **kwargs):
    """DataFrame.mode(self, axis=0, numeric_only=False)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)

    return _self.mode(axis=_axis, numeric_only=_numeric_only), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only
    }


@generator(group='pandas', name='df.pct_change')
def gen_df_pct_change(inputs, *args, **kwargs):
    """DataFrame.pct_change(self, periods=1, fill_method='pad', limit=None, freq=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _periods = Select([1] + [inp for inp in inputs if isinstance(inp, (int, np.number))])
    _limit = Select([None] + [inp for inp in inputs if isinstance(inp, (int, np.number))])

    return _self.pct_change(periods=_periods, limit=_limit), {
        'self': _self, 'periods': _periods, 'limit': _limit
    }


@generator(group='pandas', name='df.prod')
def gen_df_prod(inputs, *args, **kwargs):
    """DataFrame.prod(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0)"""

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

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame) and validate_self(inp)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    _min_count = Select([0, 1] + [inp for inp in inputs if isinstance(inp, (int, np.number))])

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.prod(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, min_count=_min_count), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'min_count': _min_count
    }


@generator(group='pandas', name='df.quantile')
def gen_df_quantile(inputs, *args, **kwargs):
    """DataFrame.quantile(self, q=0.5, axis=0, numeric_only=True, interpolation='linear')"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _q = Select([0.5] + [inp for inp in inputs if isinstance(inp, (int, np.number, float, np.floating))])
    _numeric_only = Select([True, False], fixed_domain=True)
    _interpolation = Select(['linear', 'lower', 'higher', 'midpoint', 'nearest'])

    return _self.quantile(q=_q, axis=_axis, numeric_only=_numeric_only, interpolation=_interpolation), {
        'self': _self, 'q': _q, 'axis': _axis, 'numeric_only': _numeric_only, 'interpolation': _interpolation
    }


@generator(group='pandas', name='df.rank')
def gen_df_rank(inputs, *args, **kwargs):
    """DataFrame.rank(self, axis=0, method='average', numeric_only=None, na_option='keep', ascending=True, pct=False)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _method = Select(['average', 'min', 'max', 'first', 'dense'], fixed_domain=True)
    _na_option = Select(['keep', 'top', 'bottom'], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _ascending = Select([True, False], fixed_domain=True)
    _pct = Select([True, False], fixed_domain=True)


@generator(group='pandas', name='df.round')
def gen_df_round(inputs, *args, **kwargs):
    """DataFrame.round(self, decimals=0)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _decimals = Select([0] + [inp for inp in inputs if isinstance(inp, (int, np.number, dict, pd.Series))])

    return _self.round(decimals=_decimals), {
        'self': _self, 'decimals': _decimals
    }


@generator(group='pandas', name='df.sem')
def gen_df_sem(inputs, *args, **kwargs):
    """DataFrame.sem(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    if _axis == 0:
        _ddof = Select(list(range(0, _self.shape[0])))
    else:
        _ddof = Select(list(range(0, _self.shape[1])))

    return _self.sem(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, ddof=_ddof), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'ddof': _ddof
    }


@generator(group='pandas', name='df.skew')
def gen_df_skew(inputs, *args, **kwargs):
    """DataFrame.skew(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.skew(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.sum')
def gen_df_sum(inputs, *args, **kwargs):
    """DataFrame.sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    _min_count = Select([0, 1] + [inp for inp in inputs if isinstance(inp, (int, np.number))])

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    return _self.sum(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, min_count=_min_count), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'min_count': _min_count
    }


@generator(group='pandas', name='df.std')
def gen_df_std(inputs, *args, **kwargs):
    """DataFrame.std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    if _axis == 0:
        _ddof = Select(list(range(0, _self.shape[0])))
    else:
        _ddof = Select(list(range(0, _self.shape[1])))

    return _self.std(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, ddof=_ddof), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'ddof': _ddof
    }


@generator(group='pandas', name='df.var')
def gen_df_var(inputs, *args, **kwargs):
    """DataFrame.var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([0, 1], fixed_domain=True)
    _numeric_only = Select([None, True, False], fixed_domain=True)
    _skipna = Select([True, False], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels))

    if _axis == 0:
        _ddof = Select(list(range(0, _self.shape[0])))
    else:
        _ddof = Select(list(range(0, _self.shape[1])))

    return _self.var(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, ddof=_ddof), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'ddof': _ddof
    }

    # ------------------------------------------------------------------- #
    #  Reindexing/Selection/Label Manipulations
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.add_prefix')
def gen_df_add_prefix(inputs, *args, **kwargs):
    """DataFrame.add_prefix(self, prefix)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _prefix = Select([inp for inp in inputs if isinstance(inp, str)])

    return _self.add_prefix(_prefix), {
        'self': _self, 'prefix': _prefix
    }


@generator(group='pandas', name='df.add_suffix')
def gen_df_add_suffix(inputs, *args, **kwargs):
    """DataFrame.add_suffix(self, suffix)"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _suffix = Select([inp for inp in inputs if isinstance(inp, str)])

    return _self.add_suffix(_suffix), {
        'self': _self, 'suffix': _suffix
    }


@generator(group='pandas', name='df.align')
def gen_df_align(inputs, *args, **kwargs):
    """DataFrame.align(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None, method=None,
                       limit=None, fill_axis=0, broadcast_axis=None) """

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _other = Select([inp for inp in inputs if isinstance(inp, (pd.DataFrame, pd.Series))])
    _axis = Select([None, 0, 1], fixed_domain=True)
    _broadcast_axis = Select([None, 0, 1], fixed_domain=True)
    _join = Select(['outer', 'inner', 'left', 'right'], fixed_domain=True)

    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    return _self.align(_other, join=_join, axis=_axis, level=_level, broadcast_axis=_broadcast_axis), {
        'self': _self, 'other': _other, 'join': _join, 'axis': _axis, 'level': _level, 'broadcast_axis': _broadcast_axis
    }


@generator(group='pandas', name='df.drop')
def gen_df_drop(inputs, *args, **kwargs):
    """DataFrame.drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')"""

    _self = Select([inp for inp in inputs if isinstance(inp, pd.DataFrame)])
    _axis = Select([None, 0, 1], fixed_domain=True)

    src = _self.index if _axis == 'index' else _self.columns
    level_default = Select([True, False], fixed_domain=True)
    if level_default:
        _level = None
    else:
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)])

    label_cands = set(src.get_level_values(_level)) if _level is not None else set(src)
    _labels = list(Subset(label_cands, lengths=list(range(1, len(label_cands)))))


@generator(group='pandas', name='df.drop_duplicates')
def gen_df_drop_duplicates(inputs, *args, **kwargs):
    """DataFrame.drop_duplicates(self, subset=None, keep='first', inplace=False)"""


@generator(group='pandas', name='df.duplicated')
def gen_df_duplicated(inputs, *args, **kwargs):
    """DataFrame.duplicated(self, subset=None, keep='first')"""


@generator(group='pandas', name='df.equals')
def gen_df_equals(inputs, *args, **kwargs):
    """DataFrame.equals(self, other)"""


@generator(group='pandas', name='df.filter')
def gen_df_filter(inputs, *args, **kwargs):
    """DataFrame.filter(self, items=None, like=None, regex=None, axis=None)"""


@generator(group='pandas', name='df.first')
def gen_df_first(inputs, *args, **kwargs):
    """DataFrame.first(self, offset)"""


@generator(group='pandas', name='df.idxmax')
def gen_df_idxmax(inputs, *args, **kwargs):
    """DataFrame.idxmax(self, axis=0, skipna=True)"""


@generator(group='pandas', name='df.idxmin')
def gen_df_idxmin(inputs, *args, **kwargs):
    """DataFrame.idxmin(self, axis=0, skipna=True)"""


@generator(group='pandas', name='df.last')
def gen_df_last(inputs, *args, **kwargs):
    """DataFrame.last(self, offset)"""


@generator(group='pandas', name='df.reindex')
def gen_df_reindex(inputs, *args, **kwargs):
    """DataFrame.reindex(self, labels=None, index=None, columns=None, axis=None, method=None, copy=True, level=None,
                         fill_value=nan, limit=None, tolerance=None) """


@generator(group='pandas', name='df.reindex_like')
def gen_df_reindex_like(inputs, *args, **kwargs):
    """DataFrame.reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None)"""


@generator(group='pandas', name='df.rename')
def gen_df_rename(inputs, *args, **kwargs):
    """DataFrame.rename(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None)"""


@generator(group='pandas', name='df.reset_index')
def gen_df_reset_index(inputs, *args, **kwargs):
    """DataFrame.reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill='')"""


@generator(group='pandas', name='df.set_index')
def gen_df_set_index(inputs, *args, **kwargs):
    """DataFrame.set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False)"""


@generator(group='pandas', name='df.take')
def gen_df_take(inputs, *args, **kwargs):
    """DataFrame.take(self, indices, axis=0, convert=None, is_copy=True)"""

    # ------------------------------------------------------------------- #
    #  Missing Data Handling
    # ------------------------------------------------------------------- #
