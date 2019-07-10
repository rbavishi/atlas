import itertools

from atlas import generator
from atlas.ops import Select
from atlas.semantics import DfsSemantics, op_def


@generator
def myexample(length):
    s = ""
    for i in range(length):
        s += Select(["a", "b"])

    return s


def run():
    print("# ------------------------------------------------ #")
    print("# Atlas - A Framework for Neural-Backed Generators #")
    print("# ------------------------------------------------ #")

    for i in myexample.generate(3):
        print(i)
