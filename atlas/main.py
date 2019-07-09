import itertools
from typing import Any

from atlas import generator
from atlas.ops import Select
from atlas.semantics import DfsSemantics, op_def, Semantics


class MySemantics(DfsSemantics):
    @op_def
    def Select(self, domain: Any, context: Any = None, **kwargs):
        yield from reversed(domain)


@generator(semantics=MySemantics())
def myexample(myinput):
    s = ""
    for i in range(2):
        s += Select(["a", "b"], op_id='special')

    return s


def run():
    print("# ------------------------------------------------ #")
    print("# Atlas - A Framework for Neural-Backed Generators #")
    print("# ------------------------------------------------ #")

    c = 0
    for i in itertools.islice(myexample(1), 10):
        print(i)
        c += 1

    print(c)

    myexample.set_semantics(DfsSemantics())

    c = 0
    for i in itertools.islice(myexample(1), 10):
        print(i)
        c += 1

    print(c)
