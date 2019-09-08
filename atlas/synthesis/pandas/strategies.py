import random
from typing import Dict, Set, List, Callable

import pandas as pd
import numpy as np
from atlas.operators import OpInfo
from atlas.strategies import DfsStrategy, operator
from atlas.synthesis.pandas.dataframe_generation import DfConfig, Bags


class PandasSynthesisStrategy(DfsStrategy):
    @operator
    def SelectExternal(self, domain, dtype=None, preds: List[Callable] = None, kwargs=None, **extra):
        if preds is None:
            preds = []

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

    def init(self):
        self.generated_inputs = []

    def generate_new_external(self, dtype, op_info: OpInfo, context):
        if op_info.label is None and dtype is pd.DataFrame:
            return self.df_generator.call()

        if op_info.label is not None:
            attr = f"get_ext_{op_info.label}"
            if hasattr(self, attr):
                a = getattr(self, attr)(context=context)
                return a

        return self.Sentinel

    def Sequence_function_sequence_prediction(self, **garbage):
        yield self.func_seq

    @operator
    def Select(self, domain, **garbage):
        domain = list(domain)
        random.shuffle(domain)
        yield from domain

    @operator
    def SelectExternal(self, domain, context=None, op_info: OpInfo = None, dtype=None, preds: List[Callable] = None,
                       kwargs: Dict = None, **extra):
        if preds is None:
            preds = []

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
                val = self.generate_new_external(dtype, op_info, context)
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
        print(cond_df)
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

    def get_ext_self_df_add_like(self, context=None):
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
            new_df.columns = pd.Index(random.sample(set((list(df.columns) + list(new_df.columns))), len(new_df.columns)))
        else:
            new_df.columns = pd.MultiIndex.from_tuples(
                random.sample(set((list(df.columns) + list(new_df.columns))), len(new_df.columns)))

        return new_df

    def get_ext_fill_value_df_add_like(self, context=None):
        return round(random.uniform(-100, 100), 1)
