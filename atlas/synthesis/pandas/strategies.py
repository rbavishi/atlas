import collections
import itertools
import random
import string
from typing import Dict, Set, List, Callable, Any, Collection, Optional

import pandas as pd
import numpy as np
from atlas.operators import OpInfo, operator
from atlas.strategies import DfsStrategy
from atlas.synthesis.pandas.dataframe_generation import DfConfig, Bags
from atlas.synthesis.pandas.utils import LambdaWrapper


class PandasSynthesisStrategy(DfsStrategy):
    @operator
    def SelectExternal(self, domain, dtype=None, preds: List[Callable] = None, kwargs=None, **extra):
        if preds is None:
            preds = []

        if kwargs is None:
            kwargs = {}

        unused_intermediates: Set[int] = kwargs.get('unused_intermediates', None)
        if unused_intermediates is not None:
            #  Try to yield from these first
            yield from (i for i in domain
                        if id(i) in unused_intermediates and isinstance(i, dtype) and all(p(i) for p in preds))
            yield from (i for i in domain
                        if (id(i) not in unused_intermediates) and isinstance(i, dtype) and all(p(i) for p in preds))

        else:
            yield from (i for i in domain if isinstance(i, dtype))

        if 'default' in extra:
            yield extra['default']

    @operator
    def SelectFixed(self, domain, context=None, **extra):
        yield from domain


class PandasSequentialDataGenerationStrategy(DfsStrategy):
    """
    Data-generation for the Sequential Enumerator (OOPSLA '19)
    """

    class Sentinel:
        pass

    def __init__(self, func_seq: List[str], df_generator, max_num_inputs: int = 3):
        super().__init__()
        self.func_seq = func_seq
        self.df_generator = df_generator
        self.max_num_inputs = max_num_inputs
        self.generated_inputs = []
        self.operator_iterator_bound = 10

    def init(self):
        super().init()
        self.generated_inputs = []

    def generate_new_external(self, dtype, datagen_label: Optional[str], context):
        if datagen_label is None and dtype is pd.DataFrame:
            return self.df_generator.call()

        if datagen_label is not None:
            attr = f"get_ext_{datagen_label}"
            if hasattr(self, attr):
                a = getattr(self, attr)(context=context)
                return a

        return self.Sentinel

    def generate_random_string(self, length: int):
        return ''.join(random.choice(string.ascii_letters) for _ in range(length))

    @operator(name='Sequence', tags=["function_sequence_prediction"])
    def Sequence_func(self, **garbage):
        yield self.func_seq

    @operator
    def Select(self, domain, **garbage):
        domain = list(domain)
        random.shuffle(domain)
        yield from domain

    @operator
    def SelectFixed(self, domain, **garbage):
        domain = list(domain)
        random.shuffle(domain)
        yield from domain

    @operator
    def Subset(self, domain: Any, context: Any = None, lengths: Collection[int] = None,
               include_empty: bool = False, **kwargs):
        if lengths is None:
            lengths = range(0 if include_empty else 1, len(domain) + 1)

        lengths = list(lengths)
        random.shuffle(lengths)
        domain = list(domain)
        random.shuffle(domain)

        for l in lengths:
            yield from itertools.combinations(domain, l)

    @operator
    def OrderedSubset(self, domain: Any, context: Any = None,
                      lengths: Collection[int] = None, include_empty: bool = False, **kwargs):

        if lengths is None:
            lengths = range(0 if include_empty else 1, len(domain) + 1)

        lengths = list(lengths)
        random.shuffle(lengths)
        domain = list(domain)
        random.shuffle(domain)

        for l in lengths:
            yield from itertools.permutations(domain, l)

    @operator
    def Product(self, domain: Any, context: Any = None, **kwargs):
        domain = [list(i) for i in domain]
        for i in domain:
            random.shuffle(i)

        yield from itertools.product(*domain)

    @operator
    def Sequence(self, domain: Any, context: Any = None, max_len: int = None,
                 lengths: Collection[int] = None, **kwargs):
        if max_len is None and lengths is None:
            raise SyntaxError("Sequence requires the explicit keyword argument 'max_len' or 'lengths'")

        if max_len is not None and lengths is not None:
            raise SyntaxError("Sequence takes only *one* of the 'max_len' and 'lengths' keyword arguments")

        if max_len is not None:
            for l in range(1, max_len + 1):
                yield from itertools.product(domain, repeat=l)

        elif lengths is not None:
            for l in list(lengths):
                yield from itertools.product(domain, repeat=l)

    @operator
    def SelectExternal(self, domain, context=None, op_info: OpInfo = None, dtype=None, preds: List[Callable] = None,
                       kwargs: Dict = None, datagen_label: str = None, **extra):
        if preds is None:
            preds = []

        if kwargs is None:
            kwargs = {}

        unused_intermediates: Set[int] = kwargs.get('unused_intermediates', {})
        unused_domain = [i for i in domain
                         if id(i) in unused_intermediates and isinstance(i, dtype) and all(p(i) for p in preds)]
        used_domain = [i for i in domain
                       if id(i) not in unused_intermediates and isinstance(i, dtype) and all(p(i) for p in preds)]

        if (len(unused_domain) + len(used_domain) == 0) or (len(self.generated_inputs) < self.max_num_inputs):
            used_domain.append(self.Sentinel)

        random.shuffle(unused_domain)
        random.shuffle(used_domain)

        yield from unused_domain
        for i in used_domain:
            if i is self.Sentinel:
                val = self.generate_new_external(dtype, datagen_label, context)
                if val is self.Sentinel:
                    continue

                self.generated_inputs.append(val)
                yield val
                self.generated_inputs.pop()

            else:
                yield i

        if 'default' in extra:
            yield extra['default']

    def get_ext_input_df_isna_notna(self, context=None):
        return self.df_generator.call(DfConfig(nan_prob=0.5))

    def get_ext_values_df_isin(self, context=None):
        vals = list(context['_self'].values.flatten())
        sample_size = random.randint(1, max((len(vals) - 1), 1))
        return list(random.sample(vals, sample_size))

    def get_ext_cond_df_where_mask(self, context=None):
        df, nr, nc = context['_self'], context['num_rows'], context['num_cols']
        cond_df = self.df_generator.call(DfConfig(num_rows=nr, num_cols=nc,
                                                  value_bags=Bags.bool_bags,
                                                  index_like_columns_prob=0.0))
        cond_df.index = df.index
        cond_df.columns = df.columns
        return cond_df

    def get_ext_other_df_where_mask(self, context=None):
        df, nr, nc = context['_self'], context['num_rows'], context['num_cols']
        other_df = self.df_generator.call(DfConfig(num_rows=nr, num_cols=nc))
        other_df.index = df.index
        other_df.columns = df.columns
        return other_df

    def get_ext_self_df_query(self, context=None):
        return self.df_generator.call(DfConfig(column_levels=1))

    def get_ext_expr_df_query(self, context=None):
        df = context['_self']
        pool = []
        dtypes = df.dtypes
        for col in df:
            dtype = dtypes[col]
            vals = list(df[col])
            if ('int' in str(dtype)) or ('float' in str(dtype)):
                pool.append('{} > {}'.format(col, random.choice(vals)))
                pool.append('{} < {}'.format(col, random.choice(vals)))
                pool.append('{} == {}'.format(col, random.choice(vals)))
                pool.append('{} != {}'.format(col, random.choice(vals)))
            elif 'object' in str(dtype):
                pool.append('{} == "{}"'.format(col, random.choice(vals)))
                pool.append('{} != "{}"'.format(col, random.choice(vals)))

        sample_size = random.randint(1, min(5, len(pool)))
        sample = random.sample(pool, sample_size)
        expr = sample[0]
        for i in range(1, len(sample)):
            expr += ' {} '.format(random.choice(['and', 'or']))
            expr += sample[i]

        return expr

    def get_ext_self_df_int_and_floats(self, context=None):
        return self.df_generator.call(DfConfig(value_bags=[*Bags.int_bags, *Bags.float_bags],
                                               index_like_columns_prob=0.0))

    def get_ext_other_df_add_like(self, context=None):
        df = context['_self']
        (nr, nc) = df.shape
        v_nc = random.choice([nc, nc, nc, nc - 1, nc + 1])
        new_df = self.df_generator.call(DfConfig(num_cols=v_nc, num_rows=nr,
                                                 column_levels=df.columns.nlevels,
                                                 index_like_columns_prob=0.0,
                                                 col_prefix='i1_',
                                                 value_bags=[*Bags.int_bags, *Bags.float_bags]))
        new_df.index = df.index
        if (np.random.choice([0, 1]) == 0) and (len(new_df.columns) == nc):
            new_df.columns = df.columns
        elif df.columns.nlevels == 1:
            new_df.columns = pd.Index(
                random.sample(set((list(df.columns) + list(new_df.columns))), len(new_df.columns)))
        else:
            new_df.columns = pd.MultiIndex.from_tuples(
                random.sample(set((list(df.columns) + list(new_df.columns))), len(new_df.columns)))

        return new_df

    def get_ext_fill_value_df_add_like(self, context=None):
        return round(random.uniform(-100, 100), 1)

    def get_ext_other_df_ne_like(self, context=None):
        df = context['_self']
        (nr, nc) = df.shape
        cond = self.df_generator.call(DfConfig(num_rows=nr, num_cols=nc, value_bags=Bags.bool_bags))
        cond.columns = df.columns
        cond.index = df.index

        new_df = self.df_generator.call(DfConfig(num_rows=nr, num_cols=nc))
        new_df.columns = df.columns
        new_df.index = df.index

        return df.where(cond, new_df)

    def get_ext_other_df_combine(self, context=None):
        df = context['_self']
        (nr, nc) = df.shape
        new_df = self.df_generator.call(DfConfig(num_rows=nr, num_cols=nc,
                                                 value_bags=[*Bags.int_bags, *Bags.float_bags]))
        new_df.columns = df.columns
        new_df.index = df.index
        return new_df

    def get_ext_func_df_combine(self, context=None):
        pool = [LambdaWrapper('lambda s1, s2: s1.mask(s1 < s2, s2)'),
                LambdaWrapper('lambda s1, s2: s1.mask(s1 > s2, s2)')]

        return random.choice(pool)

    def get_ext_self_df_combine_first(self, context=None):
        return self.df_generator.call(DfConfig(nan_prob=0.2))

    def get_ext_other_df_combine_first(self, context=None):
        df = context['_self']
        (nr, nc) = df.shape
        if np.random.choice([0, 1]) == 0:
            val = self.df_generator.call(DfConfig(num_cols=nc, num_rows=nr, nan_prob=0.2))
            val.columns = df.columns
            val.index = df.index

        else:
            val = self.df_generator.call(DfConfig(index_levels=df.index.nlevels, column_levels=df.columns.nlevels,
                                                  col_prefix='i1_', nan_prob=0.2))

            if df.index.nlevels == 1:
                val.index = pd.Index(random.sample(set((list(df.index) + list(val.index))), len(val.index)))
            else:
                val.index = pd.MultiIndex.from_tuples(
                    random.sample(set((list(df.index) + list(val.index))), len(val.index)))
            if df.columns.nlevels == 1:
                val.columns = pd.Index(
                    random.sample(set((list(df.columns) + list(val.columns))), len(val.columns)))
            else:
                val.columns = pd.MultiIndex.from_tuples(
                    random.sample(set((list(df.columns) + list(val.columns))), len(val.columns)))

        return val

    def get_ext_func_df_apply(self, context=None):
        df = context['_self']
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            return

        choice = random.choice(list(numeric_cols))
        return random.choice([LambdaWrapper('lambda x: x["{}"] > 1'.format(choice)),
                              LambdaWrapper('lambda x: x["{}"] + 1'.format(choice))])

    def get_ext_self_df_all_any(self, context=None):
        return self.df_generator.call(DfConfig(value_bags=[*Bags.int_bags, *Bags.string_bags,
                                                           *Bags.bool_bags, *Bags.bool_bags]))

    def get_ext_lower_df_clip(self, context=None):
        df = context['_self']
        vals = list(filter(lambda x: isinstance(x, (int, np.integer, float, np.floating)),
                           list(df.values.flatten())))
        if len(vals) == 0:
            return

        return random.uniform(min(vals), max(vals))

    def get_ext_upper_df_clip(self, context=None):
        df = context['_self']
        vals = list(filter(lambda x: isinstance(x, (int, np.integer, float, np.floating)),
                           list(df.values.flatten())))
        _lower = context['_lower'] or min(vals)
        if len(vals) == 0:
            return

        return random.uniform(_lower, max(vals))

    def get_ext_threshold_df_clip_lower_upper(self, context=None):
        df = context['_self']
        vals = list(filter((lambda x: (not isinstance(x, str))), list(df.values.flatten())))
        return random.uniform(min(vals), max(vals))

    def get_ext_other_df_corrwith(self, context=None):
        df = context['_self']
        (nr, nc) = df.shape
        val = self.df_generator.call(DfConfig(num_rows=nr, column_levels=df.columns.nlevels, col_prefix='i1_',
                                              value_bags=[*Bags.int_bags, *Bags.float_bags]))
        val.index = df.index
        if (np.random.choice([0, 1]) == 0) and (len(val.columns) == nc):
            val.columns = df.columns
        elif df.columns.nlevels == 1:
            val.columns = pd.Index(random.sample(set((list(df.columns) + list(val.columns))), len(val.columns)))
        else:
            val.columns = pd.MultiIndex.from_tuples(
                random.sample(set((list(df.columns) + list(val.columns))), len(val.columns)))

        return val

    def get_ext_self_df_count(self, context=None):
        return self.df_generator.call(DfConfig(nan_prob=0.5))

    def get_ext_self_df_computational(self, context=None):
        return self.df_generator.call(DfConfig(nan_prob=0.1))

    def get_ext_periods_df_diff(self, context=None):
        df = context['_self']
        (nr, _) = df.shape
        return random.choice(range(1 - nr, nr))

    def get_ext_str_df_add_prefix_suffix(self, context=None):
        return self.generate_random_string(random.randint(1, 8))

    def get_ext_other_df_align(self, context=None):
        df = context['_self']
        (nr, nc) = df.shape
        val = self.df_generator.call(DfConfig(num_rows=random.choice([max((nr - 1), 1), nr, (nr + 1)]),
                                              num_cols=random.choice([max((nc - 1), 1), nc, (nc + 1)]),
                                              col_prefix='i1_',
                                              index_levels=df.index.nlevels, column_levels=df.columns.nlevels,
                                              value_bags=[*Bags.int_bags, *Bags.float_bags]))
        if (np.random.choice([0, 1]) == 0) and (len(val.index) == nr):
            val.index = df.index
        elif df.index.nlevels == 1:
            val.index = pd.Index(random.sample(set((list(df.index) + list(val.index))), len(val.index)))
        else:
            val.index = pd.MultiIndex.from_tuples(
                random.sample(set((list(df.index) + list(val.index))), len(val.index)))
        if (np.random.choice([0, 1]) == 0) and (len(val.columns) == nc):
            val.columns = df.columns
        elif df.columns.nlevels == 1:
            val.columns = pd.Index(random.sample(set((list(df.columns) + list(val.columns))), len(val.columns)))
        else:
            val.columns = pd.MultiIndex.from_tuples(
                random.sample(set((list(df.columns) + list(val.columns))), len(val.columns)))

        return val

    def get_ext_self_df_duplicate_removal(self, context=None):
        return self.df_generator.call(DfConfig(min_height=3))

    def get_ext_other_df_equals(self, context=None):
        return self.df_generator.call(DfConfig(col_prefix=random.choice(['', 'i1_'])))

    def get_ext_fill_value_df_reindex(self, context=None):
        return round(random.uniform(-100, 100), 1)

    def get_ext_labels_df_reindex(self, context=None):
        df = context['_self']
        nr, nc = df.shape
        if np.random.choice([0, 1]) == 0:
            vals = list(df.index)
            new_vals = [self.generate_random_string(random.randint(1, 8)) for i in range(nr // 2)]
            return list(random.sample(vals + new_vals, nr))

        else:
            vals = list(df.columns)
            new_vals = [self.generate_random_string(random.randint(1, 8)) for i in range(nc // 2)]
            return list(random.sample(vals + new_vals, nc))

    def get_ext_other_df_reindex_like(self, context=None):
        df = context['_self']
        val = self.df_generator.call(DfConfig(index_levels=df.index.nlevels, col_prefix=random.choice(['', 'i1_']),
                                              value_bags=[*Bags.int_bags, *Bags.float_bags]))
        if df.index.nlevels == 1:
            val.index = pd.Index(random.sample(set((list(df.index) + list(val.index))), len(val.index)))
        else:
            val.index = pd.MultiIndex.from_tuples(
                random.sample(set((list(df.index) + list(val.index))), len(val.index)))

        return val

    def get_ext_indices_df_take(self, context=None):
        df = context['_self']
        (nr, nc) = df.shape
        if np.random.choice([0, 1]) == 0:
            val = random.sample(range(nr), random.choice(range(1, (nr + 1))))
            random.shuffle(val)

        else:
            val = random.sample(range(nc), random.choice(range(1, (nc + 1))))
            random.shuffle(val)

        return val

    def get_ext_self_df_dropna_fillna(self, context=None):
        return self.df_generator.call(DfConfig(nan_prob=0.3))

    def get_ext_value_df_fillna(self, context=None):
        return round(random.uniform(-100, 100), 1)

    def get_ext_self_df_pivot(self, context=None):
        return self.df_generator.call(DfConfig(column_levels=1))

    def get_ext_self_df_reorder_levels(self, context=None):
        return self.df_generator.call(DfConfig(multi_index_prob=0.6, multi_col_index_prob=0.4))

    def get_ext_right_df_merge(self, context=None):
        df = context['_self']
        new_df: pd.DataFrame = self.df_generator.call(DfConfig(col_prefix='i1_'))
        dg1 = collections.defaultdict(list)
        dg2 = collections.defaultdict(list)

        for (k, v) in dict(df.dtypes).items():
            dg1[v].append(k)
        for (k, v) in dict(new_df.dtypes).items():
            dg2[v].append(k)

        c = (set(dg1.keys()) & set(dg2.keys()))
        for dt in c:
            cols1 = list(dg1[dt])
            cols2 = list(dg2[dt])
            random.shuffle(cols1)
            random.shuffle(cols2)
            pairs = list(zip(cols1, cols2))
            for pair in pairs:
                if np.random.choice([0, 1]) == 0:
                    new_df[pair[1]] = random.sample((list(new_df[pair[1]]) + list(df[pair[0]])),
                                                    new_df.shape[0])
                    if (np.random.choice([0, 1]) == 0) and (pair[0] not in new_df.columns):
                        new_df = new_df.rename({
                            pair[1]: pair[0],
                        }, axis=1)

        return new_df
