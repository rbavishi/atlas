import itertools
from typing import List

from atlas.synthesis.pandas.dataframe_generation import generate_random_dataframe
from atlas.synthesis.pandas.engine import sequential_enumerator
from atlas.synthesis.pandas.strategies import PandasSequentialDataGenerationStrategy


def generate_sequential_data(func_seq: List[str], max_attempts: int = 10):
    strategy = PandasSequentialDataGenerationStrategy(func_seq, generate_random_dataframe)

    for _ in range(max_attempts):
        try:
            return next(iter(sequential_enumerator.with_env(strategy=strategy).generate([], None,
                                                                                        allow_unused_intermediates=False)))
        except:
            continue

    raise RuntimeError("Max. attempts reached")
