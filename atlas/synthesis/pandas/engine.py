import logging

from atlas import generator
from atlas.exceptions import ExceptionAsContinue
from atlas.synthesis.pandas.checker import Checker
from atlas.synthesis.pandas.strategies import PandasSynthesisStrategy
from atlas.utils import get_group_by_name
from atlas.synthesis.pandas.stubs import *
import atlas.synthesis.pandas.api
from typing import List, Set

api_gens = {
    gen.name: gen for gen in get_group_by_name('pandas')
}

for v in api_gens.values():
    v.caching = True


@generator(name='pandas_sequential_enumerator', strategy=PandasSynthesisStrategy(),
           caching=True)
def sequential_enumerator(inputs, output,
                          log_errors: bool = True,
                          allow_unused_intermediates: bool = True):
    """
    First decides the function sequence to explore, and then decides arguments for each function individually.
    This is the enumerator used in the OOPSLA '19 system.
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

        for obj in args.values():
            unused_intermediates.discard(id(obj))

        if (not allow_unused_intermediates) and idx == len(func_seq) and len(unused_intermediates) != 0:
            raise ExceptionAsContinue

        #  Using `id(val)` only works because DfsStrategy (and therefore PandasSynthesisStrategy) caches results
        #  at the generator call level. Therefore it won't recompute the `val` above. If this is not the case, in the
        #  subsequent generator runs, the `val` computed above will be a different object than the one in the previous
        #  run, even though the arguments may be the same.
        unused_intermediates.add(id(val))
        intermediates.append(val)
        func_args.append(args)

    return intermediates[-1], intermediates, func_seq, func_args


def solve(enumerator, inputs, output):
    checker = Checker.get_checker(output)
    for val, prog in enumerator.generate(inputs, output):
        if checker(output, val):
            print(prog)
