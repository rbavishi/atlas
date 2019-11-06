import typing

import dateutil
import pandas as pd
import numpy as np
from typing import Callable

from atlas import generator
from atlas.synthesis.pandas.stubs import Select, Subset, OrderedSubset, Product, SelectExternal, SelectFixed
from atlas.synthesis.pandas.utils import check_nan


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
def gen_df_index(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.index, {'self': _self}


@generator(group='pandas', name='df.columns')
def gen_df_columns(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.columns, {'self': _self}


@generator(group='pandas', name='df.dtypes')
def gen_df_dtypes(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.dtypes, {'self': _self}


@generator(group='pandas', name='df.ftypes')
def gen_df_ftypes(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.ftypes, {'self': _self}


@generator(group='pandas', name='df.values')
def gen_df_values(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.values, {'self': _self}


@generator(group='pandas', name='df.axes')
def gen_df_axes(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.axes, {'self': _self}


@generator(group='pandas', name='df.ndim')
def gen_df_ndim(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.ndim, {'self': _self}


@generator(group='pandas', name='df.size')
def gen_df_size(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.size, {'self': _self}


@generator(group='pandas', name='df.shape')
def gen_df_shape(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.shape, {'self': _self}


@generator(group='pandas', name='df.T')
def gen_df_T(inputs, output, *args, **kwargs):
    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.T, {'self': _self}


# ----------------------------------------------------------------------- #
#  Methods
# ----------------------------------------------------------------------- #

# ------------------------------------------------------------------- #
#  Attributes & Underlying Data
# ------------------------------------------------------------------- #


@generator(group='pandas', name='df.as_matrix')
def gen_df_as_matrix(inputs, output, *args, **kwargs):
    """DataFrame.as_matrix(self, columns=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    choose_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    if choose_default:
        _columns = None
    else:
        _columns = list(OrderedSubset(_self.columns, context=c, kwargs=kwargs, uid="3"))

    return _self.as_matrix(columns=_columns), {'self': _self, 'columns': _columns}


@generator(group='pandas', name='df.get_dtype_counts')
def gen_df_get_dtype_counts(inputs, output, *args, **kwargs):
    """DataFrame.get_dtype_counts(self)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.get_dtype_counts(), {'self': _self}


@generator(group='pandas', name='df.get_ftype_counts')
def gen_df_get_ftype_counts(inputs, output, *args, **kwargs):
    """DataFrame.get_ftype_counts(self)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    return _self.get_ftype_counts(), {'self': _self}


@generator(group='pandas', name='df.select_dtypes')
def gen_df_select_dtypes(inputs, output, *args, **kwargs):
    """DataFrame.select_dtypes(self, include=None, exclude=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}
    dtypes = set(map(str, _self.dtypes))
    use_include = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    if use_include:
        _include = Subset(dtypes, context=c, kwargs=kwargs, uid="3")
        _exclude = None
    else:
        _include = None
        _exclude = Subset(dtypes, context=c, kwargs=kwargs, uid="4")

    return _self.select_dtypes(include=_include, exclude=_exclude), {
        'self': _self, 'include': _include, 'exclude': _exclude
    }

    # ------------------------------------------------------------------- #
    #  Conversion
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.astype')
def gen_df_astype(inputs, output, *args, **kwargs):
    """DataFrame.astype(self, dtype, copy=True, errors='raise')"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    cand_dtypes = {np.dtype('int64'), np.dtype('int32'), np.dtype('float64'), np.dtype('float32'),
                   np.dtype('bool'), np.dtype('uint32'), np.dtype('uint64'), int, float, str, bool}
    cand_dtypes = set(filter(lambda x: not any(x == i for i in _self.dtypes), cand_dtypes))

    if isinstance(output, pd.DataFrame):
        cand_dtypes.update(list(output.dtypes))

    _dtype = Select(cand_dtypes, context=c, kwargs=kwargs, uid="2")

    return _self.astype(dtype=_dtype, errors='ignore'), {
        'self': _self, 'dtype': _dtype, 'errors': 'ignore'
    }


@generator(group='pandas', name='df.isna')
def gen_df_isna(inputs, output, *args, **kwargs):
    """DataFrame.isna(self)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, datagen_label="input_df_isna_notna", kwargs=kwargs, uid="1")

    return _self.isna(), {
        'self': _self
    }


@generator(group='pandas', name='df.notna')
def gen_df_notna(inputs, output, *args, **kwargs):
    """DataFrame.notna(self)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, datagen_label="input_df_isna_notna", kwargs=kwargs, uid="1")

    return _self.notna(), {
        'self': _self
    }

    # ------------------------------------------------------------------- #
    #  Indexing & Iteration
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.head')
def gen_df_head(inputs, output, *args, **kwargs):
    """DataFrame.head(self, n=5)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _n = Select(list(range(1, _self.shape[0])), context=c, kwargs=kwargs, uid="2")

    return _self.head(n=_n), {
        'self': _self, 'n': _n
    }


@generator(group='pandas', name='df.tail')
def gen_df_tail(inputs, output, *args, **kwargs):
    """DataFrame.tail(self, n=5)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _n = Select(list(range(1, _self.shape[0])), context=c, kwargs=kwargs, uid="2")

    return _self.tail(n=_n), {
        'self': _self, 'n': _n
    }


@generator(group='pandas', name='df.at.__getitem__')
def gen_df_at_getitem(inputs, output, *args, **kwargs):
    """DataFrame.at.__getitem__(self, key)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _key = (Select(_self.index, context=c, kwargs=kwargs, uid="2"),
            Select(_self.columns, context=c, kwargs=kwargs, uid="3"))

    return _self.at[_key], {
        'self': _self, 'key': _key
    }


@generator(group='pandas', name='df.iat.__getitem__')
def gen_df_iat_getitem(inputs, output, *args, **kwargs):
    """DataFrame.iat.__getitem__(self, key)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _key = (Select(list(range(_self.shape[0])), context=c, kwargs=kwargs, uid="2"),
            Select(list(range(_self.shape[1])), context=c, kwargs=kwargs, uid="3"))

    return _self.iat[_key], {
        'self': _self, 'key': _key
    }


@generator(group='pandas', name='df.loc.__getitem__')
def gen_df_loc_getitem(inputs, output, *args, **kwargs):
    """DataFrame.loc.__getitem__(self, key)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    idx_reversed = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    col_reversed = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")
    idx_start, idx_end = Subset(list(_self.index), lengths=[2], context=c, kwargs=kwargs, uid="4")
    col_start, col_end = Subset(list(_self.columns), lengths=[2], context=c, kwargs=kwargs, uid="5")

    _key = (
        (slice(idx_start, idx_end, 1) if not idx_reversed else slice(idx_end, idx_start, -1)),
        (slice(col_start, col_end, 1) if not col_reversed else slice(col_end, col_start, -1)),
    )

    return _self.loc[_key], {
        'self': _self, 'key': _key
    }


@generator(group='pandas', name='df.iloc.__getitem__')
def gen_df_iloc_getitem(inputs, output, *args, **kwargs):
    """DataFrame.iloc.__getitem__(self, key)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    idx_reversed = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    col_reversed = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")
    idx_start, idx_end = Subset(list(range(_self.shape[0])), lengths=[2], context=c, kwargs=kwargs, uid="4")
    col_start, col_end = Subset(list(range(_self.shape[1])), lengths=[2], context=c, kwargs=kwargs, uid="5")

    _key = (
        (slice(idx_start, idx_end, 1) if not idx_reversed else slice(idx_end, idx_start, -1)),
        (slice(col_start, col_end, 1) if not col_reversed else slice(col_end, col_start, -1)),
    )

    return _self.iloc[_key], {
        'self': _self, 'key': _key
    }


@generator(group='pandas', name='df.lookup')
def gen_df_lookup(inputs, output, *args, **kwargs):
    """DataFrame.lookup(self, row_labels, col_labels)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _row_labels = list(OrderedSubset(_self.index,
                                     lengths=list(range(1, min(_self.shape[0], _self.shape[1]) + 1)),
                                     context=c, kwargs=kwargs, uid="2"))
    _col_labels = list(OrderedSubset(_self.columns, lengths=[len(_row_labels)], context=c, kwargs=kwargs, uid="3"))

    return _self.lookup(row_labels=_row_labels, col_labels=_col_labels), {
        'self': _self, 'row_labels': _row_labels, 'col_labels': _col_labels
    }


@generator(group='pandas', name='df.xs')
def gen_df_xs(inputs, output, *args, **kwargs):
    """DataFrame.xs(self, key, axis=0, level=None, drop_level=True)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _drop_level = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="3")

    src = _self.index if _axis == 0 else _self.columns
    if src.nlevels == 1:
        _level = None
        _key = Select(list(src), context=c, kwargs=kwargs, uid="4")
    else:
        _level = Subset(list(range(src.nlevels)), context=c, kwargs=kwargs, uid="5")
        level_vals = [src.levels[i] for i in _level]
        _key = list(Product(level_vals, context=c, kwargs=kwargs, uid="6"))

    return _self.xs(key=_key, axis=_axis, level=_level, drop_level=_drop_level), {
        'self': _self, 'key': _key, 'axis': _axis, 'level': _level, 'drop_level': _drop_level
    }


@generator(group='pandas', name='df.isin')
def gen_df_isin(inputs, output, *args, **kwargs):
    """DataFrame.isin(self, values)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")

    #  Adding '_self' to aid data generation.
    c = {'I0': _self, 'O': output, '_self': _self}
    _values = SelectExternal(inputs, dtype=(list, tuple, pd.Series, dict, pd.DataFrame), context=c,
                             datagen_label="values_df_isin", kwargs=kwargs, uid="2")

    return _self.isin(_values), {
        'self': _self, 'values': _values
    }


@generator(group='pandas', name='df.where')
def gen_df_where(inputs, output, *args, **kwargs):
    """DataFrame.where(self, cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False,
                       raise_on_error=None)"""

    def is_valid_cond(cond):
        if isinstance(cond, pd.DataFrame) and \
                len(cond.select_dtypes(include=[bool, np.dtype('bool')]).columns) != len(cond.columns):
            return False

        return True

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")

    #  Adding num_rows and num_cols to aid data generation.
    c = {'I0': _self, 'O': output, '_self': _self, 'num_rows': _self.shape[0], 'num_cols': _self.shape[1]}

    _cond = SelectExternal(inputs, dtype=(typing.Sequence, pd.DataFrame, Callable), uid="2",
                           preds=[is_valid_cond], context=c, datagen_label="cond_df_where_mask", kwargs=kwargs)
    _other = SelectExternal(inputs, dtype=(typing.Sequence, pd.DataFrame, Callable), kwargs=kwargs, context=c,
                            datagen_label="other_df_where_mask", uid="3")

    return _self.where(_cond, other=_other, errors='ignore'), {
        'self': _self, 'cond': _cond, 'other': _other, 'errors': 'ignore'
    }


@generator(group='pandas', name='df.mask')
def gen_df_mask(inputs, output, *args, **kwargs):
    """DataFrame.mask(self, cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False,
                       raise_on_error=None)"""

    def is_valid_cond(cond):
        if isinstance(cond, pd.DataFrame) and \
                len(cond.select_dtypes(include=[bool, np.dtype('bool')]).columns) != len(cond.columns):
            return False

        return True

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")

    #  Adding num_rows and num_cols to aid data generation.
    c = {'I0': _self, 'O': output, '_self': _self, 'num_rows': _self.shape[0], 'num_cols': _self.shape[1]}

    _cond = SelectExternal(inputs, dtype=(typing.Sequence, pd.DataFrame, Callable), uid="2",
                           preds=[is_valid_cond], context=c, datagen_label="cond_df_where_mask", kwargs=kwargs)
    _other = SelectExternal(inputs, dtype=(typing.Sequence, pd.DataFrame, Callable), kwargs=kwargs, context=c,
                            datagen_label="other_df_where_mask", uid="3")

    return _self.mask(_cond, other=_other, errors='ignore'), {
        'self': _self, 'cond': _cond, 'other': _other, 'errors': 'ignore'
    }


@generator(group='pandas', name='df.query')
def gen_df_query(inputs, output, *args, **kwargs):
    """DataFrame.query(self, expr, inplace=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1",
                           datagen_label="self_df_query")

    #  Adding '_self' to aid data generation.
    c = {'I0': _self, 'O': output, '_self': _self}
    _expr = SelectExternal(inputs, dtype=str, kwargs=kwargs, uid="2",
                           context=c, datagen_label="expr_df_query")

    return _self.query(_expr), {
        'self': _self, 'expr': _expr
    }


@generator(group='pandas', name='df.__getitem__')
def gen_df_getitem(inputs, output, *args, **kwargs):
    """DataFrame.__getitem__(self, key)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    single_col = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    if single_col:
        _key = Select(_self.columns, context=c, kwargs=kwargs, uid="3")
    else:
        _key = list(OrderedSubset(_self.columns, context=c, kwargs=kwargs, uid="4"))

    return _self.__getitem__(_key), {
        'self': _self, 'key': _key
    }

    # ------------------------------------------------------------------- #
    #  Binary Operator Functions
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.add')
def gen_df_add(inputs, output, *args, **kwargs):
    """DataFrame.add(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1",
                           datagen_label="self_df_int_and_floats")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs, uid="2",
                            datagen_label="other_df_add_like")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.add(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.sub')
def gen_df_sub(inputs, output, *args, **kwargs):
    """DataFrame.sub(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1",
                           datagen_label="self_df_int_and_floats")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.sub(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.mul')
def gen_df_mul(inputs, output, *args, **kwargs):
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

    _self = SelectExternal(inputs, dtype=pd.DataFrame, preds=[validate_self], kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs, preds=[validate_other],
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.mul(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.div')
def gen_df_div(inputs, output, *args, **kwargs):
    """DataFrame.div(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.div(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.truediv')
def gen_df_truediv(inputs, output, *args, **kwargs):
    """DataFrame.truediv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.truediv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.floordiv')
def gen_df_floordiv(inputs, output, *args, **kwargs):
    """DataFrame.floordiv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.floordiv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.mod')
def gen_df_mod(inputs, output, *args, **kwargs):
    """DataFrame.mod(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.mod(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.pow')
def gen_df_pow(inputs, output, *args, **kwargs):
    """DataFrame.pow(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.pow(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.radd')
def gen_df_radd(inputs, output, *args, **kwargs):
    """DataFrame.radd(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.radd(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rsub')
def gen_df_rsub(inputs, output, *args, **kwargs):
    """DataFrame.rsub(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.rsub(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rmul')
def gen_df_rmul(inputs, output, *args, **kwargs):
    """DataFrame.rmul(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.rmul(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rdiv')
def gen_df_rdiv(inputs, output, *args, **kwargs):
    """DataFrame.rdiv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.rdiv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rtruediv')
def gen_df_rtruediv(inputs, output, *args, **kwargs):
    """DataFrame.rtruediv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.rtruediv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rfloordiv')
def gen_df_rfloordiv(inputs, output, *args, **kwargs):
    """DataFrame.rfloordiv(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.rfloordiv(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rmod')
def gen_df_rmod(inputs, output, *args, **kwargs):
    """DataFrame.rmod(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.rmod(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.rpow')
def gen_df_rpow(inputs, output, *args, **kwargs):
    """DataFrame.rpow(self, other, axis='columns', level=None, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    _fill_value = SelectExternal(inputs, dtype=(pd.DataFrame, int, float, np.floating, np.integer), default=None,
                                 datagen_label="fill_value_df_add_like", kwargs=kwargs, context=c, uid="6")

    return _self.rpow(other=_other, axis=_axis, level=_level, fill_value=_fill_value), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.lt')
def gen_df_lt(inputs, output, *args, **kwargs):
    """DataFrame.lt(self, other, axis='columns', level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    return _self.lt(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.gt')
def gen_df_gt(inputs, output, *args, **kwargs):
    """DataFrame.gt(self, other, axis='columns', level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    return _self.gt(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.le')
def gen_df_le(inputs, output, *args, **kwargs):
    """DataFrame.le(self, other, axis='columns', level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    return _self.le(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.ge')
def gen_df_ge(inputs, output, *args, **kwargs):
    """DataFrame.ge(self, other, axis='columns', level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_add_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    return _self.ge(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.ne')
def gen_df_ne(inputs, output, *args, **kwargs):
    """DataFrame.ne(self, other, axis='columns', level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_ne_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    return _self.ne(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.eq')
def gen_df_eq(inputs, output, *args, **kwargs):
    """DataFrame.eq(self, other, axis='columns', level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series, list, tuple, int, str, float),
                            context=c, kwargs=kwargs,
                            datagen_label="other_df_ne_like", uid="2")

    if isinstance(_other, pd.Series):
        _axis = SelectFixed(['columns', 'index'], context=c, kwargs=kwargs, uid="3")
    else:
        _axis = 'columns'

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 'index' else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    return _self.eq(other=_other, axis=_axis, level=_level), {
        'self': _self, 'other': _other, 'axis': _axis, 'level': _level
    }


@generator(group='pandas', name='df.combine')
def gen_df_combine(inputs, output, *args, **kwargs):
    """DataFrame.combine(self, other, func, fill_value=None, overwrite=True)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_int_and_floats", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=pd.DataFrame, context=c, kwargs=kwargs,
                            datagen_label="other_df_combine", uid="2")
    _func = SelectExternal(inputs, dtype=Callable, kwargs=kwargs, datagen_label="func_df_combine", uid="3")

    _overwrite = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    fill_val_cands = [inp for inp in inputs if np.isscalar(inp)]
    if len(fill_val_cands) > 0:
        _fill_value = Select([None] + fill_val_cands, context=c, kwargs=kwargs, uid="5")
    else:
        _fill_value = None

    return _self.combine(other=_other, func=_func, fill_value=_fill_value, overwrite=_overwrite), {
        'self': _self, 'other': _other, 'func': _func, 'fill_value': _fill_value, 'overwrite': _overwrite
    }


@generator(group='pandas', name='df.combine_first')
def gen_df_combine_first(inputs, output, *args, **kwargs):
    """DataFrame.combine_first(self, other)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_combine_first", uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, context=c,
                            datagen_label="other_df_combine_first", uid="2")

    return _self.combine_first(other=_other), {
        'self': _self, 'other': _other
    }

    # ------------------------------------------------------------------- #
    #  Function application, GroupBy & Window
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.apply')
def gen_df_apply(inputs, output, *args, **kwargs):
    """DataFrame.apply(self, func, axis=0, broadcast=False, raw=False, reduce=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _func = SelectExternal(inputs, dtype=Callable, kwargs=kwargs, context=c, datagen_label="func_df_apply", uid="2")
    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="3")
    _broadcast = SelectFixed([False, True], context=c, kwargs=kwargs, uid="4")
    _raw = SelectFixed([False, True], context=c, kwargs=kwargs, uid="5")

    return _self.apply(func=_func, axis=_axis, broadcast=_broadcast, raw=_raw), {
        'self': _self, 'func': _func, 'axis': _axis, 'broadcast': _broadcast, 'raw': _raw
    }


@generator(group='pandas', name='df.applymap', metadata={'data-generation': False})
def gen_df_applymap(inputs, output, *args, **kwargs):
    """DataFrame.applymap(self, func)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _func = SelectExternal(inputs, dtype=Callable, kwargs=kwargs, context=c, uid="2")

    return _self.applymap(func=_func), {
        'self': _self, 'func': _func
    }


@generator(group='pandas', name='df.agg', metadata={'data-generation': False})
def gen_df_agg(inputs, output, *args, **kwargs):
    """DataFrame.agg(self, func, axis=0)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _func = SelectExternal(inputs, dtype=(str, dict, list, tuple, Callable), kwargs=kwargs,
                           context=c, uid="2")
    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="3")

    return _self.agg(func=_func), {
        'self': _self, 'func': _func, 'axis': _axis
    }


@generator(group='pandas', name='df.transform', metadata={'data-generation': False})
def gen_df_transform(inputs, output, *args, **kwargs):
    """DataFrame.transform(self, func)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _func = SelectExternal(inputs, dtype=(str, dict, list, tuple, Callable), kwargs=kwargs, context=c, uid="2")

    return _self.transform(func=_func), {
        'self': _self, 'func': _func
    }


@generator(group='pandas', name='df.groupby')
def gen_df_groupby(inputs, output, *args, **kwargs):
    """DataFrame.groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _sort = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")
    _as_index = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        single = SelectFixed([True, False], context=c, kwargs=kwargs, uid="6")
        if single:
            _level = Select(list(range(0, src.nlevels - 1)), context=c, kwargs=kwargs, uid="7")
        else:
            _level = list(OrderedSubset(list(range(src.nlevels)),
                                        lengths=list(range(2, src.nlevels + 1)), context=c, kwargs=kwargs, uid="8"))

    if _level is not None:
        _by = None
    else:
        use_ext = SelectFixed([True, False], context=c, kwargs=kwargs, uid="9")
        if use_ext:
            dimension = _self.shape[0] if _axis == 0 else _self.shape[1]
            _by = Select([inp for inp in inputs
                          if isinstance(inp, (pd.Series, list, tuple, dict, np.ndarray)) and len(inp) == dimension],
                         context=c, kwargs=kwargs, uid="10")

        else:
            cols = list(_self.columns)
            index = _self.index
            index_cols = [index.names[i] for i in range(index.nlevels) if index.names[i] is not None]
            _by = list(Subset(cols + list(index_cols), context=c, kwargs=kwargs, uid="11"))

    return _self.groupby(by=_by, axis=_axis, level=_level, as_index=_as_index, sort=_sort), {
        'self': _self, 'by': _by, 'axis': _axis, 'level': _level, 'as_index': _as_index, 'sort': _sort
    }

    # ------------------------------------------------------------------- #
    #  Computations/Descriptive Stats
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.abs')
def gen_df_abs(inputs, output, *args, **kwargs):
    """DataFrame.abs(self)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_int_and_floats", uid="1")
    return _self.abs(), {
        'self': _self
    }


@generator(group='pandas', name='df.all')
def gen_df_all(inputs, output, *args, **kwargs):
    """DataFrame.all(self, axis=None, bool_only=None, skipna=None, level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_all_any", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _bool_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    return _self.all(axis=_axis, bool_only=_bool_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'bool_only': _bool_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.any')
def gen_df_any(inputs, output, *args, **kwargs):
    """DataFrame.any(self, axis=None, bool_only=None, skipna=None, level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_all_any", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _bool_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    return _self.any(axis=_axis, bool_only=_bool_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'bool_only': _bool_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.clip')
def gen_df_clip(inputs, output, *args, **kwargs):
    """DataFrame.clip(self, lower=None, upper=None, axis=None, inplace=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_int_and_floats", uid="1")
    c = {'I0': _self, 'O': output, '_self': _self}

    upper_cands = [inp for inp in inputs if isinstance(inp, (float, np.floating, int, np.integer))]
    lower_default = None
    upper_default = None

    if isinstance(output, pd.DataFrame):
        try:
            lower_default = np.min(output.select_dtypes(include=np.number).values)
            upper_default = np.max(output.select_dtypes(include=np.number).values)
        except:
            pass

    _lower = SelectExternal(inputs, dtype=(float, np.floating, int, np.integer), default=lower_default,
                            kwargs=kwargs, context=c, datagen_label="lower_df_clip", uid="2")

    c['_lower'] = _lower
    _upper = SelectExternal(inputs, dtype=(float, np.floating, int, np.integer), default=upper_default,
                            kwargs=kwargs, context=c, datagen_label="upper_df_clip", uid="3")

    return _self.clip(lower=_lower, upper=_upper), {
        'self': _self, 'lower': _lower, 'upper': _upper
    }


@generator(group='pandas', name='df.clip_lower')
def gen_df_clip_lower(inputs, output, *args, **kwargs):
    """DataFrame.clip_lower(self, threshold, axis=None, inplace=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_int_and_floats", uid="1")
    c = {'I0': _self, 'O': output, '_self': _self}

    default_threshold = None
    if isinstance(output, pd.DataFrame):
        try:
            default_threshold = np.min(output.select_dtypes(include=np.number).values)
        except:
            pass

    _threshold = SelectExternal(inputs, dtype=(float, np.floating, int, np.number), kwargs=kwargs, uid="2",
                                context=c, datagen_label="threshold_df_clip_lower_upper", default=default_threshold)

    return _self.clip_lower(threshold=_threshold), {
        'self': _self, 'threshold': _threshold
    }


@generator(group='pandas', name='df.clip_upper')
def gen_df_clip_upper(inputs, output, *args, **kwargs):
    """DataFrame.clip_upper(self, threshold, axis=None, inplace=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_int_and_floats", uid="1")
    c = {'I0': _self, 'O': output, '_self': _self}

    default_threshold = None
    if isinstance(output, pd.DataFrame):
        try:
            default_threshold = np.max(output.select_dtypes(include=np.number).values)
        except:
            pass

    _threshold = SelectExternal(inputs, dtype=(float, np.floating, int, np.number), kwargs=kwargs, uid="2",
                                context=c, datagen_label="threshold_df_clip_lower_upper", default=default_threshold)

    return _self.clip_upper(threshold=_threshold), {
        'self': _self, 'threshold': _threshold
    }


@generator(group='pandas', name='df.corr')
def gen_df_corr(inputs, output, *args, **kwargs):
    """DataFrame.corr(self, method='pearson', min_periods=1)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _min_periods = Select([1] + [inp for inp in inputs if isinstance(inp, (int, np.number))],
                          context=c, kwargs=kwargs, uid="2")
    _method = SelectFixed(['pearson', 'kendall', 'spearman'], context=c, kwargs=kwargs, uid="3")

    return _self.corr(min_periods=_min_periods, method=_method), {
        'self': _self, 'min_periods': _min_periods, 'method': _method
    }


@generator(group='pandas', name='df.corrwith')
def gen_df_corrwith(inputs, output, *args, **kwargs):
    """DataFrame.corrwith(self, other, axis=0, drop=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_int_and_floats", uid="1")
    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, context=c,
                            datagen_label="other_df_corrwith", uid="2")

    _drop = SelectFixed([False, True], context=c, kwargs=kwargs, uid="3")
    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="4")

    return _self.corrwith(_other, axis=_axis, drop=_drop), {
        'self': _self, 'other': _other, 'axis': _axis, 'drop': _drop
    }


@generator(group='pandas', name='df.count')
def gen_df_count(inputs, output, *args, **kwargs):
    """DataFrame.count(self, axis=0, level=None, numeric_only=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_count", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([False, True], context=c, kwargs=kwargs, uid="3")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="5")

    return _self.count(axis=_axis, level=_level, numeric_only=_numeric_only), {
        'self': _self, 'axis': _axis, 'level': _level, 'numeric_only': _numeric_only
    }


@generator(group='pandas', name='df.cov')
def gen_df_cov(inputs, output, *args, **kwargs):
    """DataFrame.cov(self, min_periods=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _min_periods = SelectExternal(inputs, dtype=(int, np.integer), default=None, kwargs=kwargs, context=c, uid="2")

    return _self.cov(min_periods=_min_periods), {
        'self': _self, 'min_periods': _min_periods
    }


@generator(group='pandas', name='df.cummax')
def gen_df_cummax(inputs, output, *args, **kwargs):
    """DataFrame.cummax(self, axis=None, skipna=True)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")

    return _self.cummax(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.cummin')
def gen_df_cummin(inputs, output, *args, **kwargs):
    """DataFrame.cummin(self, axis=None, skipna=True)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")

    return _self.cummin(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.cumprod')
def gen_df_cumprod(inputs, output, *args, **kwargs):
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

    _self = SelectExternal(inputs, dtype=pd.DataFrame, preds=[validate_self], kwargs=kwargs,
                           datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")

    return _self.cumprod(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.cumsum')
def gen_df_cumsum(inputs, output, *args, **kwargs):
    """DataFrame.cumsum(self, axis=None, skipna=True)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")

    return _self.cumsum(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.diff')
def gen_df_diff(inputs, output, *args, **kwargs):
    """DataFrame.diff(self, periods=1, axis=0)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output, '_self': _self}
    _periods = SelectExternal(inputs, dtype=(int, np.integer), default=1, kwargs=kwargs, context=c,
                              datagen_label="periods_df_diff", uid="2")

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="3")

    return _self.diff(axis=_axis, periods=_periods), {
        'self': _self, 'axis': _axis, 'periods': _periods
    }


@generator(group='pandas', name='df.eval', metadata={'data-generation': False})
def gen_df_eval(inputs, output, *args, **kwargs):
    """DataFrame.eval(self, expr, inplace=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _expr = SelectExternal(inputs, dtype=str, kwargs=kwargs, context=c, uid="2")

    return _self.eval(_expr), {
        'self': _self, 'expr': _expr
    }


@generator(group='pandas', name='df.kurt')
def gen_df_kurt(inputs, output, *args, **kwargs):
    """DataFrame.kurt(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    return _self.kurt(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.mad')
def gen_df_mad(inputs, output, *args, **kwargs):
    """DataFrame.mad(self, axis=None, skipna=None, level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="5"))

    return _self.mad(axis=_axis, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.max')
def gen_df_max(inputs, output, *args, **kwargs):
    """DataFrame.max(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    return _self.max(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.mean')
def gen_df_mean(inputs, output, *args, **kwargs):
    """DataFrame.max(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    return _self.mean(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.median')
def gen_df_median(inputs, output, *args, **kwargs):
    """DataFrame.median(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    return _self.median(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.min')
def gen_df_min(inputs, output, *args, **kwargs):
    """DataFrame.min(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    return _self.min(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.mode')
def gen_df_mode(inputs, output, *args, **kwargs):
    """DataFrame.mode(self, axis=0, numeric_only=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")

    return _self.mode(axis=_axis, numeric_only=_numeric_only), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only
    }


@generator(group='pandas', name='df.pct_change')
def gen_df_pct_change(inputs, output, *args, **kwargs):
    """DataFrame.pct_change(self, periods=1, fill_method='pad', limit=None, freq=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _periods = Select([1] + [inp for inp in inputs if isinstance(inp, (int, np.number))], context=c, kwargs=kwargs, uid="2")
    _limit = Select([None] + [inp for inp in inputs if isinstance(inp, (int, np.number))], context=c, kwargs=kwargs, uid="3")

    return _self.pct_change(periods=_periods, limit=_limit), {
        'self': _self, 'periods': _periods, 'limit': _limit
    }


@generator(group='pandas', name='df.prod')
def gen_df_prod(inputs, output, *args, **kwargs):
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

    _self = SelectExternal(inputs, dtype=pd.DataFrame, preds=[validate_self], kwargs=kwargs,
                           datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    _min_count = Select([0, 1] + [inp for inp in inputs if isinstance(inp, (int, np.number))], context=c, kwargs=kwargs, uid="5")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="6")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="7"))

    return _self.prod(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, min_count=_min_count), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'min_count': _min_count
    }


@generator(group='pandas', name='df.quantile')
def gen_df_quantile(inputs, output, *args, **kwargs):
    """DataFrame.quantile(self, q=0.5, axis=0, numeric_only=True, interpolation='linear')"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _q = Select([0.5] + [inp for inp in inputs
                         if isinstance(inp, (int, np.number, float, np.floating, typing.Sequence))], context=c, kwargs=kwargs, uid="3")
    _numeric_only = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    _interpolation = Select(['linear', 'lower', 'higher', 'midpoint', 'nearest'], context=c, kwargs=kwargs, uid="5")

    return _self.quantile(q=_q, axis=_axis, numeric_only=_numeric_only, interpolation=_interpolation), {
        'self': _self, 'q': _q, 'axis': _axis, 'numeric_only': _numeric_only, 'interpolation': _interpolation
    }


@generator(group='pandas', name='df.rank')
def gen_df_rank(inputs, output, *args, **kwargs):
    """DataFrame.rank(self, axis=0, method='average', numeric_only=None, na_option='keep', ascending=True, pct=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _method = SelectFixed(['average', 'min', 'max', 'first', 'dense'], context=c, kwargs=kwargs, uid="3")
    _na_option = SelectFixed(['keep', 'top', 'bottom'], context=c, kwargs=kwargs, uid="4")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="5")
    _ascending = SelectFixed([True, False], context=c, kwargs=kwargs, uid="6")
    _pct = SelectFixed([True, False], context=c, kwargs=kwargs, uid="7")

    return _self.rank(axis=_axis, method=_method, numeric_only=_numeric_only,
                      na_option=_na_option, ascending=_ascending, pct=_pct), {
               'self': _self, 'axis': _axis, 'method': _method, 'numeric_only': _numeric_only, 'na_option': _na_option,
               'ascending': _ascending, 'pct': _pct
           }


@generator(group='pandas', name='df.round')
def gen_df_round(inputs, output, *args, **kwargs):
    """DataFrame.round(self, decimals=0)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _decimals = Select([0] + [inp for inp in inputs if isinstance(inp, (int, np.number, dict, pd.Series))],
                       context=c, kwargs=kwargs, uid="2")

    return _self.round(decimals=_decimals), {
        'self': _self, 'decimals': _decimals
    }


@generator(group='pandas', name='df.sem')
def gen_df_sem(inputs, output, *args, **kwargs):
    """DataFrame.sem(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    if _axis == 0:
        _ddof = Select(list(range(0, max(_self.shape[0], 2))), context=c, kwargs=kwargs, uid="7")
    else:
        _ddof = Select(list(range(0, max(_self.shape[1], 2))), context=c, kwargs=kwargs, uid="8")

    return _self.sem(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, ddof=_ddof), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'ddof': _ddof
    }


@generator(group='pandas', name='df.skew')
def gen_df_skew(inputs, output, *args, **kwargs):
    """DataFrame.skew(self, axis=None, skipna=None, level=None, numeric_only=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    return _self.skew(axis=_axis, numeric_only=_numeric_only, skipna=_skipna, level=_level), {
        'self': _self, 'axis': _axis, 'numeric_only': _numeric_only, 'skipna': _skipna, 'level': _level
    }


@generator(group='pandas', name='df.sum')
def gen_df_sum(inputs, output, *args, **kwargs):
    """DataFrame.sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")

    _min_count = Select([0, 1] + [inp for inp in inputs if isinstance(inp, (int, np.number))], context=c, kwargs=kwargs, uid="5")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="6")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="7"))

    return _self.sum(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, min_count=_min_count), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'min_count': _min_count
    }


@generator(group='pandas', name='df.std')
def gen_df_std(inputs, output, *args, **kwargs):
    """DataFrame.std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    if _axis == 0:
        _ddof = Select(list(range(0, max(_self.shape[0], 2))), context=c, kwargs=kwargs, uid="7")
    else:
        _ddof = Select(list(range(0, max(_self.shape[1], 2))), context=c, kwargs=kwargs, uid="7")

    return _self.std(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, ddof=_ddof), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'ddof': _ddof
    }


@generator(group='pandas', name='df.var')
def gen_df_var(inputs, output, *args, **kwargs):
    """DataFrame.var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_computational", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _numeric_only = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="3")
    _skipna = SelectFixed([None, True, False], context=c, kwargs=kwargs, uid="4")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        levels = [(src.names[i] or i) for i in range(src.nlevels)]
        _level = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="6"))

    if _axis == 0:
        _ddof = Select(list(range(0, max(_self.shape[0], 2))), context=c, kwargs=kwargs, uid="7")
    else:
        _ddof = Select(list(range(0, max(_self.shape[1], 2))), context=c, kwargs=kwargs, uid="7")

    return _self.var(axis=_axis, numeric_only=True, skipna=_skipna, level=_level, ddof=_ddof), {
        'self': _self, 'axis': _axis, 'numeric_only': True, 'skipna': _skipna, 'level': _level, 'ddof': _ddof
    }

    # ------------------------------------------------------------------- #
    #  Reindexing/Selection/Label Manipulations
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.add_prefix')
def gen_df_add_prefix(inputs, output, *args, **kwargs):
    """DataFrame.add_prefix(self, prefix)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _prefix = SelectExternal(inputs, dtype=str, kwargs=kwargs, datagen_label="str_df_add_prefix_suffix",
                             context=c, uid="2")

    return _self.add_prefix(_prefix), {
        'self': _self, 'prefix': _prefix
    }


@generator(group='pandas', name='df.add_suffix')
def gen_df_add_suffix(inputs, output, *args, **kwargs):
    """DataFrame.add_suffix(self, suffix)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _suffix = SelectExternal(inputs, dtype=str, kwargs=kwargs, datagen_label="str_df_add_prefix_suffix",
                             context=c, uid="2")

    return _self.add_suffix(_suffix), {
        'self': _self, 'suffix': _suffix
    }


@generator(group='pandas', name='df.align')
def gen_df_align(inputs, output, *args, **kwargs):
    """DataFrame.align(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None, method=None,
                       limit=None, fill_axis=0, broadcast_axis=None) """

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=(pd.DataFrame, pd.Series), kwargs=kwargs, context=c,
                            datagen_label="other_df_align", uid="2")

    _axis = SelectFixed([None, 0, 1], context=c, kwargs=kwargs, uid="3")
    _broadcast_axis = SelectFixed([None, 0, 1], context=c, kwargs=kwargs, uid="4")
    _join = SelectFixed(['outer', 'inner', 'left', 'right'], context=c, kwargs=kwargs, uid="5")

    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="6")
    if level_default:
        _level = None
    else:
        src = _self.index if _axis == 0 else _self.columns
        _level = list(OrderedSubset([(src.names[i] or i) for i in range(src.nlevels)],
                                    context=c, kwargs=kwargs, uid="7"))

    return _self.align(_other, join=_join, axis=_axis, level=_level, broadcast_axis=_broadcast_axis), {
        'self': _self, 'other': _other, 'join': _join, 'axis': _axis, 'level': _level, 'broadcast_axis': _broadcast_axis
    }


@generator(group='pandas', name='df.drop')
def gen_df_drop(inputs, output, *args, **kwargs):
    """DataFrame.drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([None, 0, 1], context=c, kwargs=kwargs, uid="2")

    src = _self.index if _axis == 0 else _self.columns
    level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")
    if level_default:
        _level = None
    else:
        _level = Select([(src.names[i] or i) for i in range(src.nlevels)], context=c, kwargs=kwargs, uid="4")

    label_cands = set(src.get_level_values(_level)) if _level is not None else set(src)
    _labels = list(Subset(label_cands, lengths=list(range(1, len(label_cands))), context=c, kwargs=kwargs, uid="5"))

    return _self.drop(labels=_labels, axis=_axis, level=_level, errors='ignore'), {
        'self': _self, 'labels': _labels, 'axis': _axis, 'level': _level, 'errors': 'ignore'
    }


@generator(group='pandas', name='df.drop_duplicates')
def gen_df_drop_duplicates(inputs, output, *args, **kwargs):
    """DataFrame.drop_duplicates(self, subset=None, keep='first', inplace=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_duplicate_removal", uid="1")
    c = {'I0': _self, 'O': output}

    _subset = list(Subset(_self.columns, context=c, kwargs=kwargs, uid="2"))
    _keep = SelectFixed(['first', 'last', False], context=c, kwargs=kwargs, uid="3")

    return _self.drop_duplicates(subset=_subset, keep=_keep), {
        'self': _self, 'subset': _subset, 'keep': _keep
    }


@generator(group='pandas', name='df.duplicated')
def gen_df_duplicated(inputs, output, *args, **kwargs):
    """DataFrame.duplicated(self, subset=None, keep='first')"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                           datagen_label="self_df_duplicate_removal", uid="1")
    c = {'I0': _self, 'O': output}

    _subset = list(Subset(_self.columns, context=c, kwargs=kwargs, uid="2"))
    _keep = SelectFixed(['first', 'last', False], context=c, kwargs=kwargs, uid="3")

    return _self.duplicated(subset=_subset, keep=_keep), {
        'self': _self, 'subset': _subset, 'keep': _keep
    }


@generator(group='pandas', name='df.equals')
def gen_df_equals(inputs, output, *args, **kwargs):
    """DataFrame.equals(self, other)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    _other = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="other_df_equals", uid="2")

    return _self.equals(_other), {
        'self': _self, 'other': _other
    }


@generator(group='pandas', name='df.filter')
def gen_df_filter(inputs, output, *args, **kwargs):
    """DataFrame.filter(self, items=None, like=None, regex=None, axis=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    mode = SelectFixed(['use_items', 'use_like', 'use_regex'], context=c, kwargs=kwargs, uid="2")
    if mode == 'use_items':
        _items = list(Subset(_self.columns, context=c, kwargs=kwargs, uid="3"))
        return _self.filter(items=_items), {
            'self': _self, 'items': _items, 'like': None, 'regex': None, 'axis': None
        }

    elif mode == 'use_like':
        _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="4")
        _like = SelectExternal(inputs, dtype=str, kwargs=kwargs, context=c, uid="5")
        return _self.filter(like=_like, axis=_axis), {
            'self': _self, 'like': _like, 'axis': _axis, 'items': None, 'regex': None
        }

    else:
        _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="6")
        _regex = SelectExternal(inputs, dtype=str, kwargs=kwargs, context=c, uid="7")
        return _self.filter(regex=_regex, axis=_axis), {
            'self': _self, 'regex': _regex, 'axis': _axis, 'items': None, 'like': None
        }


@generator(group='pandas', name='df.first', metadata={'data-generation': False})
def gen_df_first(inputs, output, *args, **kwargs):
    """DataFrame.first(self, offset)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _offset = Select([inp for inp in inputs
                      if isinstance(inp, (str, pd.DateOffset, dateutil.relativedelta.relativedelta))],
                     context=c, kwargs=kwargs, uid="2")

    return _self.first(_offset), {
        'self': _self, 'offset': _offset
    }


@generator(group='pandas', name='df.idxmax')
def gen_df_idxmax(inputs, output, *args, **kwargs):
    """DataFrame.idxmax(self, axis=0, skipna=True)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_int_and_floats", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")

    return _self.idxmax(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.idxmin')
def gen_df_idxmin(inputs, output, *args, **kwargs):
    """DataFrame.idxmin(self, axis=0, skipna=True)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_int_and_floats", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _skipna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="3")

    return _self.idxmin(axis=_axis, skipna=_skipna), {
        'self': _self, 'axis': _axis, 'skipna': _skipna
    }


@generator(group='pandas', name='df.last', metadata={'data-generation': False})
def gen_df_last(inputs, output, *args, **kwargs):
    """DataFrame.last(self, offset)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _offset = Select([inp for inp in inputs
                      if isinstance(inp, (str, pd.DateOffset, dateutil.relativedelta.relativedelta))],
                     context=c, kwargs=kwargs, uid="2")

    return _self.last(_offset), {
        'self': _self, 'offset': _offset
    }


@generator(group='pandas', name='df.reindex')
def gen_df_reindex(inputs, output, *args, **kwargs):
    """DataFrame.reindex(self, labels=None, index=None, columns=None, axis=None, method=None, copy=True, level=None,
                         fill_value=nan, limit=None, tolerance=None) """

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output, '_self': _self}

    _labels = SelectExternal(inputs, dtype=(tuple, list), datagen_label="labels_df_reindex",
                             kwargs=kwargs, default=None, context=c, uid="2")
    _fill_value = SelectExternal(inputs, dtype=object, preds=[np.isscalar], datagen_label="fill_value_df_reindex",
                                 kwargs=kwargs, default=np.NaN, context=c, uid="3")
    _limit = SelectExternal(inputs, dtype=(int, np.integer), default=None, kwargs=kwargs, context=c, uid="4")

    if isinstance(output, pd.DataFrame) and SelectFixed([True, False], context=c, kwargs=kwargs, uid="5"):
        return _self.reindex(index=output.index, columns=output.columns, limit=_limit, fill_value=_fill_value), {
            'self': _self, 'labels': None, 'index': output.index, 'columns': output.columns, 'limit': _limit,
            'fill_value': _fill_value,
        }

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="6")
    src = _self.index if _axis == 0 else _self.columns
    if src.nlevels == 1:
        _level = None
    else:
        _level = Select([(src.names[i] or i) for i in range(0, src.nlevels)], context=c, kwargs=kwargs, uid="7")

    return _self.reindex(labels=_labels, axis=_axis, level=_level, fill_value=_fill_value, limit=_limit), {
        'self': _self, 'labels': _labels, 'axis': _axis, 'level': _level, 'fill_value': _fill_value, 'limit': _limit
    }


@generator(group='pandas', name='df.reindex_like')
def gen_df_reindex_like(inputs, output, *args, **kwargs):
    """DataFrame.reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output, '_self': _self}
    _other = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs,
                            context=c, datagen_label="other_df_reindex_like", uid="2")

    _method = SelectFixed([None, 'bfill', 'pad', 'nearest'], context=c, kwargs=kwargs, uid="3")

    return _self.reindex_like(_other, method=_method), {
        'self': _self, 'other': _other, 'method': _method
    }


@generator(group='pandas', name='df.rename', metadata={'data-generation': False})
def gen_df_rename(inputs, output, *args, **kwargs):
    """DataFrame.rename(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    use_index_columns = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    if use_index_columns:
        _index = SelectExternal(inputs, dtype=(dict, Callable), kwargs=kwargs, context=c, uid="3")
        _columns = SelectExternal(inputs, dtype=(dict, Callable), kwargs=kwargs, context=c, uid="4")

        return _self.rename(index=_index, columns=_columns), {
            'self': _self, 'index': _index, 'columns': _columns
        }

    else:
        _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="5")
        _mapper = SelectExternal(inputs, dtype=(dict, Callable), kwargs=kwargs, context=c, uid="6")
        src = _self.index if _axis == 0 else _self.columns
        if src.nlevels == 1:
            _level = None
        else:
            _level = Select([(src.names[i] or i) for i in range(0, src.nlevels)], context=c, kwargs=kwargs, uid="7")

        return _self.rename(axis=_axis, mapper=_mapper, level=_level), {
            'self': _self, 'mapper': _mapper, 'axis': _axis, 'level': _level
        }


@generator(group='pandas', name='df.reset_index')
def gen_df_reset_index(inputs, output, *args, **kwargs):
    """DataFrame.reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill='')"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _drop = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    level_default = not (_self.index.nlevels > 1 and SelectFixed([True, False], context=c, kwargs=kwargs, uid="3"))
    if level_default:
        _level = None
    else:
        index = _self.index
        levels = [(index.names[i] or i) for i in range(index.nlevels)]
        _level = list(Subset(levels, lengths=list(range(1, index.nlevels)), context=c, kwargs=kwargs, uid="4"))

    col_level_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if col_level_default:
        _col_level = 0
    else:
        colindex = _self.columns
        _col_level = Select([(colindex.names[i] or i) for i in range(1, colindex.nlevels)], context=c, kwargs=kwargs, uid="6")

    _col_fill = Select([None] + [inp for inp in inputs if isinstance(inp, str)], context=c, kwargs=kwargs, uid="7")

    return _self.reset_index(level=_level, drop=_drop, col_level=_col_level, col_fill=_col_fill), {
        'self': _self, 'level': _level, 'drop': _drop, 'col_level': _col_level, 'col_fill': _col_fill
    }


@generator(group='pandas', name='df.set_index')
def gen_df_set_index(inputs, output, *args, **kwargs):
    """DataFrame.set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _drop = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    _append = SelectFixed([False, True], context=c, kwargs=kwargs, uid="3")
    _keys = list(OrderedSubset(_self.columns, lengths=list(range(1, len(_self.columns))), context=c, kwargs=kwargs, uid="4"))

    return _self.set_index(keys=_keys, drop=_drop, append=_append), {
        'self': _self, 'keys': _keys, 'drop': _drop, 'append': _append
    }


@generator(group='pandas', name='df.take')
def gen_df_take(inputs, output, *args, **kwargs):
    """DataFrame.take(self, indices, axis=0, convert=None, is_copy=True)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")

    c = {'I0': _self, 'O': output, '_self': _self}
    _indices = SelectExternal(inputs, dtype=typing.Sequence, kwargs=kwargs, context=c,
                              datagen_label="indices_df_take", uid="2")

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="3")

    return _self.take(indices=_indices, axis=_axis), {
        'self': _self, 'indices': _indices, 'axis': _axis
    }

    # ------------------------------------------------------------------- #
    #  Missing Data Handling
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.dropna')
def gen_df_dropna(inputs, output, *args, **kwargs):
    """DataFrame.dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_dropna_fillna", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _how = SelectFixed(['any', 'all'], context=c, kwargs=kwargs, uid="3")

    default_subset = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    if default_subset:
        _subset = None
    else:
        src = _self.columns if _axis == 0 else _self.index
        _subset = list(Subset(src, lengths=list(range(1, len(src))), context=c, kwargs=kwargs, uid="5"))

    _thresh = Select([None] + [inp for inp in inputs if isinstance(inp, (int, np.number))], context=c, kwargs=kwargs, uid="6")

    return _self.dropna(axis=_axis, how=_how, thresh=_thresh, subset=_subset), {
        'self': _self, 'axis': _axis, 'how': _how, 'thresh': _thresh, 'subset': _subset
    }


@generator(group='pandas', name='df.fillna')
def gen_df_fillna(inputs, output, *args, **kwargs):
    """DataFrame.fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_dropna_fillna", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([None, 0, 1], context=c, kwargs=kwargs, uid="2")
    _method = SelectFixed([None, 'backfill', 'bfill', 'pad', 'ffill'], context=c, kwargs=kwargs, uid="3")
    nans = max(sum(1 for i in _self.values if check_nan(i)), _self.count().sum())
    _limit = Select([None] + list(range(1, nans + 1)), context=c, kwargs=kwargs, uid="4")

    value_default = (_method is not None) and SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if value_default:
        _value = None
    else:
        value_cands = list(inputs)
        if isinstance(output, (pd.DataFrame, pd.Series)):
            value_cands.extend(output.values.flatten())

        _value = SelectExternal(value_cands, dtype=object, preds=[np.isscalar], kwargs=kwargs,
                                datagen_label="value_df_fillna", context=c, uid="6")

    return _self.fillna(value=_value, method=_method, axis=_axis, limit=_limit), {
        'self': _self, 'value': _value, 'method': _method, 'axis': _axis, 'limit': _limit
    }

    # ------------------------------------------------------------------- #
    #  Reshaping, Sorting, Transposing
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.pivot_table')
def gen_df_pivot_table(inputs, output, *args, **kwargs):
    """DataFrame.pivot_table(self, values=None, index=None, columns=None, aggfunc='mean', fill_value=None,
                             margins=False, dropna=True, margins_name='All')"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    if _self.index.nlevels == 1 and _self.columns.nlevels == 1:
        _margins = SelectFixed([False, True], context=c, kwargs=kwargs, uid="2")
    else:
        _margins = False

    _aggfunc = Select(set(['mean', 'sum', 'min', 'max', 'median'] + [inp for inp in inputs
                                                                     if isinstance(inp, (Callable, tuple, str))]),
                      context=c, kwargs=kwargs, uid="3")

    _fill_value = Select([None] + [inp for inp in inputs if np.isscalar(inp)], context=c, kwargs=kwargs, uid="4")
    _dropna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    _margins_name = Select(['All'] + [inp for inp in inputs if isinstance(inp, str)], context=c, kwargs=kwargs, uid="6")

    columns_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="7")
    if columns_default:
        _columns = []
    else:
        _columns = list(OrderedSubset(_self.columns, context=c, kwargs=kwargs, uid="8"))

    index_default = SelectFixed([True, False], context=c, kwargs=kwargs, uid="9")
    if index_default:
        _index = []
    else:
        _index = OrderedSubset(set(_self.columns) - set(_columns), context=c, kwargs=kwargs, uid="10")

    #  Check if aggfunc works on non-numeric stuff
    try:
        agg = {
            'mean': np.mean, 'sum': np.sum, 'min': np.min, 'max': np.max, 'median': np.median
        }[_aggfunc]
        _ = agg(pd.Series(['a', 'b']))
        works = True

    except:
        works = False

    if not works:
        columns = list(set(_self.select_dtypes(include=np.number).columns) - set(_columns) - set(_index))
    else:
        columns = list(set(_self.columns) - set(_columns) - set(_index))

    col_domain = [col for col in columns if not isinstance(col, (list, tuple))]

    singleton = SelectFixed([True, False], context=c, kwargs=kwargs, uid="11")
    if singleton:
        _values = Select(col_domain, context=c, kwargs=kwargs, uid="12")
    else:
        _values = list(OrderedSubset(columns, context=c, kwargs=kwargs, uid="13"))

    return _self.pivot_table(values=_values, index=_index, columns=_columns, aggfunc=_aggfunc, fill_value=_fill_value,
                             margins=_margins, dropna=_dropna, margins_name=_margins_name), {
               'self': _self, 'values': _values, 'index': _index, 'columns': _columns, 'aggfunc': _aggfunc,
               'fill_value': _fill_value, 'margins': _margins, 'dropna': _dropna, 'margins_name': _margins_name
           }


@generator(group='pandas', name='df.pivot')
def gen_df_pivot(inputs, output, *args, **kwargs):
    """DataFrame.pivot(self, index=None, columns=None, values=None)"""

    def dup_filter(cand):
        try:
            return not any(_self[[cand, _columns]].duplicated())
        except:
            return True

    _self = SelectExternal(inputs, dtype=pd.DataFrame, datagen_label='self_df_pivot', kwargs=kwargs, uid="1")

    c = {'I0': _self, 'O': output}
    _columns = Select(_self.columns, context=c, kwargs=kwargs, uid="2")
    _index = Select([None] + list(filter(dup_filter, set(_self.columns) - {_columns})), context=c, kwargs=kwargs, uid="3")
    if _self.index.nlevels > 1 and _index is None:
        _values = None
    else:
        _values = Select(set(_self.columns) | {None}, context=c, kwargs=kwargs, uid="4")

    return _self.pivot(columns=_columns, index=_index, values=_values), {
        'self': _self, 'columns': _columns, 'index': _index, 'values': _values
    }


@generator(group='pandas', name='df.reorder_levels')
def gen_df_reorder_levels(inputs, output, *args, **kwargs):
    """DataFrame.reorder_levels(self, order, axis=0)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, datagen_label="self_df_reorder_levels", uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    src = _self.index if _axis == 0 else _self.columns
    levels = [(src.names[i] or i) for i in range(src.nlevels)]
    _order = list(OrderedSubset(levels, context=c, kwargs=kwargs, uid="3"))

    return _self.reorder_levels(order=_order, axis=_axis), {
        'self': _self, 'order': _order, 'axis': _axis
    }


@generator(group='pandas', name='df.sort_values')
def gen_df_sort_values(inputs, output, *args, **kwargs):
    """DataFrame.sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _axis = SelectFixed([0, 1], context=c, kwargs=kwargs, uid="2")
    _na_position = SelectFixed(['last', 'first'], context=c, kwargs=kwargs, uid="3")

    if _axis == 0:
        _by = list(OrderedSubset(list(_self.columns) + [i for i in _self.index.names if i is not None],
                                 context=c, kwargs=kwargs, uid="4"))
    else:
        _by = list(OrderedSubset(list(_self.index), context=c, kwargs=kwargs, uid="5"))

    _ascending = SelectFixed([True, False], context=c, kwargs=kwargs, uid="6")

    return _self.sort_values(by=_by, axis=_axis, ascending=_ascending, na_position=_na_position), {
        'self': _self, 'by': _by, 'axis': _axis, 'ascending': _ascending, 'na_position': _na_position
    }


@generator(group='pandas', name='df.stack')
def gen_df_stack(inputs, output, *args, **kwargs):
    """DataFrame.stack(self, level=-1, dropna=True)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _dropna = SelectFixed([True, False], context=c, kwargs=kwargs, uid="2")
    level_default = not (_self.columns.nlevels > 1 and SelectFixed([True, False], context=c, kwargs=kwargs, uid="3"))
    if level_default:
        _level = -1
    else:
        columns = _self.columns
        levels = [(columns.names[i] or i) for i in range(columns.nlevels)]
        _level = list(OrderedSubset(levels, lengths=list(range(1, columns.nlevels + 1)), context=c, kwargs=kwargs, uid="4"))

    return _self.stack(level=_level, dropna=_dropna), {
        'self': _self, 'level': _level, 'dropna': _dropna
    }


@generator(group='pandas', name='df.unstack')
def gen_df_unstack(inputs, output, *args, **kwargs):
    """DataFrame.unstack(self, level=-1, fill_value=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    level_default = not (_self.index.nlevels > 1 and SelectFixed([True, False], context=c, kwargs=kwargs, uid="2"))
    if level_default:
        _level = -1
    else:
        index = _self.index
        levels = [(index.names[i] or i) for i in range(index.nlevels)]
        _level = list(OrderedSubset(levels, lengths=list(range(1, index.nlevels + 1)), context=c, kwargs=kwargs, uid="3"))

    fill_value_cands = {inp for inp in inputs if np.isscalar(inp)}
    if isinstance(output, (pd.Series, pd.DataFrame)):
        fill_value_cands.update(output.values.flatten())

    fill_value_default = not (len(fill_value_cands) > 0 and SelectFixed([True, False], context=c, kwargs=kwargs, uid="4"))
    if fill_value_default:
        _fill_value = None
    else:
        _fill_value = Select(fill_value_cands, context=c, kwargs=kwargs, uid="5")

    return _self.unstack(level=_level, fill_value=_fill_value), {
        'self': _self, 'level': _level, 'fill_value': _fill_value
    }


@generator(group='pandas', name='df.melt')
def gen_df_melt(inputs, output, *args, **kwargs):
    """DataFrame.melt(self, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _var_name = Select([None] + [inp for inp in inputs if isinstance(inp, str)], context=c, kwargs=kwargs, uid="2")
    _value_name = Select(['value'] + [inp for inp in inputs if isinstance(inp, str)], context=c, kwargs=kwargs, uid="3")

    default_id_vars = SelectFixed([True, False], context=c, kwargs=kwargs, uid="4")
    default_value_vars = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")

    if default_id_vars:
        _id_vars = None
    else:
        _id_vars = list(OrderedSubset(_self.columns, context=c, kwargs=kwargs, uid="6"))

    if default_value_vars:
        _value_vars = None
    else:
        _value_vars = list(OrderedSubset(list(set(_self.columns) - set(_id_vars or [])), context=c, kwargs=kwargs, uid="7"))

    col_level_default = not (_self.columns.nlevels > 1 and SelectFixed([True, False], context=c, kwargs=kwargs, uid="8"))
    if col_level_default:
        _col_level = None
    else:
        _col_level = Select(list(range(0, _self.columns.nlevels)), context=c, kwargs=kwargs, uid="9")

    return _self.melt(id_vars=_id_vars, value_vars=_value_vars, var_name=_var_name,
                      value_name=_value_name, col_level=_col_level), {
               'self': _self, 'id_vars': _id_vars, 'value_vars': _value_vars, 'var_name': _var_name,
               'value_name': _value_name,
               'col_level': _col_level
           }

    # ------------------------------------------------------------------- #
    #  Combining/Joining/Merging
    # ------------------------------------------------------------------- #


@generator(group='pandas', name='df.merge')
def gen_df_merge(inputs, output, *args, **kwargs):
    """DataFrame.merge(self, right, how='inner', on=None, left_on=None, right_on=None,
    `                  left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'),
                       copy=True, indicator=False, validate=None)"""

    _self = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output, '_self': _self}
    _right = SelectExternal(inputs, dtype=pd.DataFrame, kwargs=kwargs, context=c,
                            datagen_label="right_df_merge", uid="2")

    c['I1'] = _right
    _how = SelectFixed(['inner', 'outer', 'left', 'right'], context=c, kwargs=kwargs, uid="3")
    _sort = SelectFixed([False, True], context=c, kwargs=kwargs, uid="4")

    use_on = SelectFixed([True, False], context=c, kwargs=kwargs, uid="5")
    if use_on:
        common_cols = set(_self.columns) & set(_right.columns)
        _on = list(Subset(common_cols, context=c, kwargs=kwargs, uid="6"))

        return _self.merge(right=_right, how=_how, on=_on, sort=_sort), {
            'self': _self, 'right': _right, 'how': _how, 'on': _on, 'sort': _sort
        }

    else:
        _left_index = SelectFixed([False, True], context=c, kwargs=kwargs, uid="7")
        _right_index = SelectFixed([False, True], context=c, kwargs=kwargs, uid="8")

        _left_on = None
        _right_on = None

        if not _left_index:
            #  Cannot use left_on if left_index is activated
            columns = set(_self.columns)
            lengths = None
            if _right_index:
                lengths = [_right.index.nlevels]

            _left_on = list(Subset(columns, lengths=lengths, context=c, kwargs=kwargs, uid="9"))

        if not _right_index:
            # Cannot use right_on if right_index is activated
            columns = set(_right.columns)
            lengths = None
            if _left_index:
                lengths = [_self.index.nlevels]
            else:
                lengths = [len(_left_on)]

            _right_on = list(OrderedSubset(columns, lengths=lengths, context=c, kwargs=kwargs, uid="10"))

        return _self.merge(_right, how=_how, sort=_sort, left_index=_left_index, right_index=_right_index,
                           left_on=_left_on, right_on=_right_on), {
                   'self': _self, 'right': _right, 'how': _how, 'sort': _sort, 'left_index': _left_index,
                   'right_index': _right_index, 'left_on': _left_on, 'right_on': _right_on
               }


# ======================================================================= #
# ======================================================================= #
#                        The DataFrameGroupBy API
# ======================================================================= #
# ======================================================================= #

pd_dfgroupby = pd.core.groupby.DataFrameGroupBy


@generator(group='pandas', name='dfgroupby.count')
def gen_dfgroupby_count(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.count(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.count(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.first')
def gen_dfgroupby_first(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.first(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.first(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.last')
def gen_dfgroupby_last(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.last(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.last(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.max')
def gen_dfgroupby_max(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.max(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.max(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.mean')
def gen_dfgroupby_mean(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.mean(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.mean(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.median')
def gen_dfgroupby_median(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.median(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.median(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.min')
def gen_dfgroupby_min(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.min(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.min(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.idxmin')
def gen_dfgroupby_idxmin(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.idxmin(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.idxmin(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.idxmax')
def gen_dfgroupby_idxmax(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.idxmax(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.idxmax(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.prod')
def gen_dfgroupby_prod(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.prod(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.prod(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.size')
def gen_dfgroupby_size(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.size(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.size(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.sum')
def gen_dfgroupby_sum(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.sum(self)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    return _self.sum(), {
        'self': _self
    }


@generator(group='pandas', name='dfgroupby.transform')
def gen_dfgroupby_transform(inputs, output, *args, **kwargs):
    """DataFrameGroupBy.transform(self, func)"""

    _self = SelectExternal(inputs, dtype=pd_dfgroupby, kwargs=kwargs, uid="1")
    c = {'I0': _self, 'O': output}

    _func = SelectExternal(inputs, dtype=(str, dict, list, tuple, Callable), kwargs=kwargs, context=c, uid="2")
    return _self.transform(func=_func), {
        'self': _self, 'func': _func
    }
