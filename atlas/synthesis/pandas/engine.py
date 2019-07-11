from atlas import generator
from atlas.synthesis.pandas.checker import Checker
from atlas.utils import get_group_by_name
from atlas.stubs import Select, Sequences, Subsets, OrderedSubsets
from atlas.synthesis.pandas.api import *

api_gens = {
    gen.name: gen for gen in get_group_by_name('pandas')
}


@generator(group='pandas', name='pandas_synthesis_brute_force')
def brute_force_enumerator(inputs, output):
    func = api_gens[Select(list(api_gens.keys()))]
    return func.name, func(inputs)


def solve(inputs, output):
    checker = Checker.get_checker(output)
    for fname, (val, prog) in brute_force_enumerator.generate(inputs, output):
        if checker(output, val):
            print(fname, prog)
