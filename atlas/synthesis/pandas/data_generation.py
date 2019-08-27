from typing import List
from atlas import generator
from atlas.exceptions import ExceptionAsContinue
from atlas.synthesis.pandas.api import PandasDataGenerationStrategy
from atlas.synthesis.pandas.dataframe_generation import generate_random_dataframe

from atlas.utils import get_group_by_name

api_gens = {
    gen.name: gen for gen in get_group_by_name('pandas')
}


@generator(strategy=PandasDataGenerationStrategy(generate_random_dataframe))
def generate_function_arguments(inputs, func_seq: List[str]):
    intermediates = []
    prog = []
    for func_str in func_seq:
        func = api_gens[func_str]
        try:
            val, args = func(intermediates + inputs, output=None)
        except Exception as e:
            print(e)
            raise ExceptionAsContinue

        prog.append((func_str, args))
        intermediates.append(val)

    return intermediates[-1], prog
