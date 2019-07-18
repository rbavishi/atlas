import unittest
from typing import Any, List

import pandas as pd
import numpy as np

import atlas.synthesis.pandas.api
from atlas import generator
from atlas.exceptions import ExceptionAsContinue
from atlas.synthesis.pandas.checker import Checker
from atlas.utils import get_group_by_name

api_gens = {
    gen.name: gen for gen in get_group_by_name('pandas')
}


@generator(group='pandas')
def simple_enumerator(inputs, output, func_seq):
    prog = []
    intermediates = []
    for func in func_seq:
        func = api_gens[func]
        try:
            val, args = func(intermediates + inputs, output)
        except:
            raise ExceptionAsContinue
        prog.append((func.name, args))
        intermediates.append(val)

    return intermediates[-1]


class TestGenerators(unittest.TestCase):
    def check(self, inputs: List[Any], output: Any, funcs: List[str], seqs: List[List[int]],
              constants: List[Any] = None):
        if constants is not None:
            inputs += constants

        checker = Checker.get_checker(output)
        func_seqs = [[funcs[i] for i in seq] for seq in seqs]
        for func_seq in func_seqs:
            for val in simple_enumerator.generate(inputs, output, func_seq):
                if checker(output, val):
                    return True

        print(inputs, output)
        self.assertTrue(False, "Did not find a solution")

    def test_df_index(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].index
        funcs = ['df.index']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_index_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = [0, 1, 2, 3]
        funcs = ['df.index']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_columns(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].columns
        funcs = ['df.columns']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_columns_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = ['A', 'B', 'C']
        funcs = ['df.columns']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_dtypes(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].dtypes
        funcs = ['df.dtypes']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_ftypes(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].ftypes
        funcs = ['df.ftypes']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_values(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].values
        funcs = ['df.values']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_axes(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].axes
        funcs = ['df.axes']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_ndim(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].ndim
        funcs = ['df.ndim']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_size(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].size
        funcs = ['df.size']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_shape(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].shape
        funcs = ['df.shape']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_T(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].T
        funcs = ['df.T']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_as_matrix(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].as_matrix(['A', 'C', 'B'])
        funcs = ['df.as_matrix']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_as_matrix_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].as_matrix()
        funcs = ['df.as_matrix']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_get_dtype_counts(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].get_dtype_counts()
        funcs = ['df.get_dtype_counts']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_get_ftype_counts(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].get_ftype_counts()
        funcs = ['df.get_ftype_counts']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_select_dtypes(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0][['A', 'C']]
        funcs = ['df.select_dtypes']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_astype(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0][['A', 'C']].astype('int32')
        funcs = ['df.astype', 'df.__getitem__']
        seqs = [[1, 0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_isna(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].isna()
        funcs = ['df.isna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_notna(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].notna()
        funcs = ['df.notna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_head(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].head(2)
        funcs = ['df.head']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_at_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].at[(1, 'B')]
        funcs = ['df.at.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_at_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].at[(2, 'A')]
        funcs = ['df.at.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_iat_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].iat[(1, 1)]
        funcs = ['df.iat.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_iat_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].iat[(2, 2)]
        funcs = ['df.iat.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_loc_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].loc[1:3, 'B':'A':(- 1)]
        funcs = ['df.loc.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_loc_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].loc[[1, 3], 'C':'A':(- 1)].head(1)
        funcs = ['df.loc.__getitem__', 'df.head']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_iloc_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].iloc[1:3, 1:0:(- 1)]
        funcs = ['df.iloc.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_iloc_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].iloc[[1, 3], 2:0:(- 1)].head(1)
        funcs = ['df.iloc.__getitem__', 'df.head']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_lookup(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].lookup([0, 2], ['A', 'C'])
        funcs = ['df.lookup']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_lookup_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].lookup([0, 1], ['A', 'B'])
        funcs = ['df.lookup']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_xs(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].xs(0, axis=0)
        funcs = ['df.xs']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_xs_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].xs('C', axis=1)
        funcs = ['df.xs']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_xs_3(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].xs(['one', 'bar'], level=[1, 0])
        funcs = ['df.xs']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_xs_4(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        inputs = [inputs[0].T]
        output = inputs[0].xs(['one', 'bar'], level=[1, 0], axis=1)
        funcs = ['df.xs']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_tail(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].tail(2)
        funcs = ['df.tail']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_isin(self):
        constants = [[1, 3, 12, 'a']]
        inputs = [pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'f'],
        })]
        output = inputs[0].isin([1, 3, 12, 'a'])
        funcs = ['df.isin']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_where(self):
        constants = [(lambda _df: ((_df % 3) == 0))]
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B']),
                  (- pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B']))]
        output = inputs[0].where(constants[0], (- inputs[1]))
        funcs = ['df.where']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_mask(self):
        constants = [(lambda _df: ((_df % 3) == 0))]
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B']),
                  (- pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B']))]
        output = inputs[0].mask(constants[0], (- inputs[1]))
        funcs = ['df.mask']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_query(self):
        constants = ['a > b']
        inputs = [pd.DataFrame(np.random.randn(10, 2), columns=list('ab'))]
        output = inputs[0].query('a > b')
        funcs = ['df.query']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0]['A']
        funcs = ['df.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0][['B', 'C', 'A']]
        funcs = ['df.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_add(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].add(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.add']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_sub(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].sub(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.sub']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_mul(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].mul(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.mul']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_div(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].div(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.div']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_truediv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].truediv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.truediv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_floordiv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].floordiv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.floordiv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_mod(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].mod(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.mod']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_pow(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].pow(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.pow']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_radd(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].radd(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.radd']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rsub(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rsub(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rsub']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rmul(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rmul(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rmul']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rdiv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rdiv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rdiv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rtruediv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rtruediv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rtruediv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rfloordiv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rfloordiv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rfloordiv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rmod(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rmod(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rmod']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rpow(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rpow(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rpow']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_lt(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].lt(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.lt']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_gt(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].gt(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.gt']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_le(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].le(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.le']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_ge(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].ge(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.ge']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_ne(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].ne(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.ne']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_eq(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].eq(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.eq']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_combine(self):
        constants = [(lambda s1, s2: (s1 if (s1.sum() < s2.sum()) else s2))]
        inputs = [pd.DataFrame({
            'A': [0, 0],
            'B': [4, 4],
        }), pd.DataFrame({
            'A': [1, 1],
            'B': [3, 3],
        })]
        output = inputs[0].combine(inputs[1], constants[0])
        funcs = ['df.combine']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_combine_first(self):
        inputs = [pd.DataFrame([[1, np.nan]]), pd.DataFrame([[3, 4]])]
        output = inputs[0].combine_first(inputs[1])
        funcs = ['df.combine_first']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)
