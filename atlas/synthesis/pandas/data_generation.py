import logging

import pandas as pd

from typing import List, Set

from atlas import generator
from atlas.exceptions import ExceptionAsContinue
from atlas.synthesis.pandas.dataframe_generation import generate_random_dataframe
from atlas.synthesis.pandas.strategies import PandasSequentialDataGenerationStrategy
from atlas.synthesis.pandas.stubs import *
from atlas.synthesis.pandas.utils import ThreadingTimeout
from atlas.utils import get_group_by_name
import atlas.synthesis.pandas.api


api_gens = {
    gen.name: gen for gen in get_group_by_name('pandas')
}

for v in api_gens.values():
    v.caching = True


@generator(name='pandas_sequential_enumerator', caching=True)
def sequential_enumerator(inputs, output,
                          log_errors: bool = False):
    """
    Copy of the sequential enumerator for synthesis with extra functionality to aid data-generation.
    """
    func_seq: List[str] = Sequence(list(api_gens.keys()), max_len=3, tags=['function_sequence_prediction'])
    func_args = []
    intermediates = []
    unused_intermediates: Set[int] = set()

    for idx, func in enumerate(func_seq, 1):
        func_gen = api_gens[func]
        try:
            val, args = func_gen(intermediates + inputs, output, idx=idx, unused_intermediates=unused_intermediates)
        except ExceptionAsContinue:
            raise
        except Exception as e:
            if log_errors:
                logging.exception(e)

            raise ExceptionAsContinue

        if isinstance(val, pd.DataFrame):
            if val.shape[0] > 15 or val.shape[1] > 15:
                raise ExceptionAsContinue

        elif isinstance(val, pd.Series):
            if val.shape[0] > 25:
                raise ExceptionAsContinue

        for obj in args.values():
            unused_intermediates.discard(id(obj))

        if idx == len(func_seq) and len(unused_intermediates) != 0:
            raise ExceptionAsContinue

        #  Using `id(val)` only works because DfsStrategy (and therefore PandasSynthesisStrategy) caches results
        #  at the generator call level. Therefore it won't recompute the `val` above. If this is not the case, in the
        #  subsequent generator runs, the `val` computed above will be a different object than the one in the previous
        #  run, even though the arguments may be the same.
        unused_intermediates.add(id(val))
        intermediates.append(val)
        func_args.append(args)

    return intermediates[-1], intermediates, func_seq, func_args


def generate_sequential_data(func_seq: List[str], max_attempts: int = 10, attempt_timeout: int = 10):
    strategy = PandasSequentialDataGenerationStrategy(func_seq, generate_random_dataframe)

    for _ in range(max_attempts):
        try:
            with ThreadingTimeout(attempt_timeout):
                return sequential_enumerator.with_env(strategy=strategy, tracing=True).call([], None), \
                       strategy.generated_inputs[:]
        except:
            continue

    raise RuntimeError("Max. attempts reached")
