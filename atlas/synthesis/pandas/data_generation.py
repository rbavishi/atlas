from typing import List
from atlas import generator
from atlas.exceptions import ExceptionAsContinue
from atlas.synthesis.pandas.api import PandasDataGenerationStrategy
from atlas.synthesis.pandas.dataframe_generation import generate_random_dataframe

from atlas.utils import get_group_by_name

api_gens = {
    gen.name: gen for gen in get_group_by_name('pandas')
}


def helper_generate_function_arguments(all_inputs, func: str):
    strategy = PandasDataGenerationStrategy(generate_random_dataframe)
    new_inputs = []
    for (output, args), trace in api_gens[func].generate(all_inputs, output=None,
                                                         generated_inputs=new_inputs).with_strategy(
        strategy).with_tracing():
        yield output, args, new_inputs, trace
        new_inputs = []


def generate_function_arguments(inputs: List, used_intermediates: List, unused_intermediates: List, func: str):
    #  First enumerate with unused intermediates if any
    if len(unused_intermediates) > 0:
        yield from helper_generate_function_arguments(unused_intermediates, func)

    yield from helper_generate_function_arguments(inputs + used_intermediates, func)


def helper_generate_sequence_arguments(func_seq: List[str], inputs: List,
                                       used_intermediates: List, unused_intermediates: List,
                                       current_program: List, traces: List):
    func = func_seq[0]
    unused_ids = {id(i): i for i in unused_intermediates}
    for output, args, new_inputs, trace in generate_function_arguments(inputs, used_intermediates, unused_intermediates,
                                                                       func):
        new_used = {id(i) for i in args.values() if id(i) in unused_ids}
        still_unused = [i for i in unused_intermediates if id(i) not in new_used]
        program = current_program + [(func, args)]
        if len(func_seq) == 1:
            if len(still_unused) > 0:
                continue

            yield output, program, traces + [trace]

        else:
            still_unused.append(output)
            yield from helper_generate_sequence_arguments(func_seq[1:], inputs + new_inputs,
                                                          used_intermediates + list(new_used), still_unused, program,
                                                          traces + [trace])


def generate_sequence_arguments(func_seq: List[str]):
    yield from helper_generate_sequence_arguments(func_seq, [], [], [], [], [])
