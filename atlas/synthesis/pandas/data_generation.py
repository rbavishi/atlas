import itertools
from typing import List

from atlas.synthesis.pandas.dataframe_generation import generate_random_dataframe
from atlas.synthesis.pandas.engine import sequential_enumerator
from atlas.synthesis.pandas.strategies import PandasSequentialDataGenerationStrategy


def generate_sequential_data(func_seq: List[str]):
    strategy = PandasSequentialDataGenerationStrategy(func_seq, generate_random_dataframe)
    a = []
    return next(iter(sequential_enumerator.generate([], None,
                                                    allow_unused_intermediates=False).with_strategy(strategy)))
