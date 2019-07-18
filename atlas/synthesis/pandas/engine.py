from atlas import generator
from atlas.exceptions import ExceptionAsContinue
from atlas.synthesis.pandas.checker import Checker
from atlas.utils import get_group_by_name
from atlas.stubs import Select, Sequence, Subset, OrderedSubset
import atlas.synthesis.pandas.api

api_gens = {
    gen.name: gen for gen in get_group_by_name('pandas')
}


@generator(group='pandas', name='pandas_synthesis_brute_force')
def brute_force_enumerator(inputs, output):
    func_seq = Sequence(list(api_gens.keys()), max_len=2, oid='func_seq_sequence')

    prog = []
    intermediates = []
    for func in func_seq:
        func = api_gens[func]
        try:
            val, args = func(intermediates + inputs, output)
        except:
            raise ExceptionAsContinue
        prog.append((func.name, args))
        intermediates.append(val)

    return intermediates[-1], prog


def solve(inputs, output):
    checker = Checker.get_checker(output)
    for val, prog in brute_force_enumerator.generate(inputs, output):
        if checker(output, val):
            print(prog)
