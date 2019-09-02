import random
from typing import Dict, Set, List

import pandas as pd
from atlas.operators import OpInfo
from atlas.strategies import DfsStrategy, operator
from atlas.synthesis.pandas.dataframe_generation import DfConfig


class PandasSynthesisStrategy(DfsStrategy):
    @operator
    def SelectExternal(self, domain, dtype=None, kwargs=None, **garbage):
        unused_intermediates: Set[int] = kwargs.get('unused_intermediates', None)
        if unused_intermediates is not None:
            #  Try to yield from these first
            yield from (i for i in domain if id(i) in unused_intermediates and isinstance(i, dtype))
            yield from (i for i in domain if (id(i) not in unused_intermediates) and isinstance(i, dtype))

        else:
            yield from (i for i in domain if isinstance(i, dtype))


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
    def SelectExternal(self, domain, context=None, op_info: OpInfo = None, dtype=None, kwargs: Dict = None, **garbage):
        unused_intermediates: Set[int] = kwargs.get('unused_intermediates', {})
        unused_domain = [i for i in domain if id(i) in unused_intermediates and isinstance(i, dtype)]
        used_domain = [i for i in domain if id(i) not in unused_intermediates and isinstance(i, dtype)]

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

    def get_ext_input_df_isna_notna(self, context=None):
        return self.df_generator.call(DfConfig(nan_prob=0.5))

    def get_ext_values_df_isin(self, context=None):
        vals = list(context['_self'].values.flatten())
        sample_size = random.randint(1, max((len(vals) - 1), 1))
        return list(random.sample(vals, sample_size))
