import itertools
from typing import List

from atlas.synthesis.pandas.dataframe_generation import generate_random_dataframe
from atlas.synthesis.pandas.engine import sequential_enumerator
from atlas.synthesis.pandas.strategies import PandasSequentialDataGenerationStrategy


def generate_sequential_data(func_seq: List[str]):
    strategy = PandasSequentialDataGenerationStrategy(func_seq, generate_random_dataframe)
    a = []
    for result in itertools.islice(sequential_enumerator.generate([], None).with_strategy(strategy), 2):
        print("GOT")
        a.append((result, strategy.generated_inputs))

    for (a1, a2, a3), a4 in a:
        print(a1, a3[1]['dtype'])
        # print("func_seq, args", a2, a3)
