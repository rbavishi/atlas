import itertools
from typing import Dict, Any, Callable, Generator, Collection

from atlas.exceptions import ExceptionAsContinue
from atlas.strategies import Strategy, op_def


class PeekableGenerator:
    def __init__(self, gen: Generator):
        self.gen = gen
        self._finished: bool = False
        self._cur_val: Any = None
        self._next_val: Any = next(gen)
        self.step()

    def is_finished(self):
        return self._finished

    def peek(self):
        return self._cur_val

    def step(self):
        self._cur_val = self._next_val

        try:
            self._next_val = next(self.gen)
        except StopIteration:
            self._finished = True


class DfsStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.call_id: int = 0
        self.op_map: Dict[int, PeekableGenerator] = {}
        self.last_unfinished: int = None

    def init(self):
        self.call_id = 0
        self.op_map = {}
        self.last_unfinished = None

    def init_run(self):
        self.call_id = 0
        self.last_unfinished = -1

    def finish_run(self):
        if self.last_unfinished > -1:
            self.op_map = {k: v for k, v in self.op_map.items() if k <= self.last_unfinished}
            self.op_map[self.last_unfinished].step()

    def is_finished(self):
        return self.last_unfinished == -1

    def make_call(self, op_kind: str, op_id: str) -> Callable:
        label = op_kind
        if op_kind + "_" + op_id in dir(self):
            label = op_kind + "_" + op_id

        handler = getattr(self, label)

        def call(domain: Any, context: Any = None, oid=op_id, **kwargs):
            t = self.call_id
            self.call_id += 1

            if t not in self.op_map:
                try:
                    op: PeekableGenerator = PeekableGenerator(handler(domain, context=context, oid=oid, **kwargs))

                except StopIteration:
                    #  Operator received an empty domain
                    raise ExceptionAsContinue

                self.op_map[t] = op

            else:
                op = self.op_map[t]

            if not op.is_finished():
                self.last_unfinished = t

            return op.peek()

        return call

    @op_def
    def Select(self, domain: Any, context: Any = None, fixed_domain=False, **kwargs):
        yield from domain

    @op_def
    def Subset(self, domain: Any, context: Any = None, lengths: Collection[int] = None,
                include_empty: bool = False, **kwargs):
        if lengths is None:
            lengths = range(0 if include_empty else 1, len(domain) + 1)

        for l in lengths:
            yield from itertools.combinations(domain, l)

    @op_def
    def OrderedSubset(self, domain: Any, context: Any = None,
                       lengths: Collection[int] = None, include_empty: bool = False, **kwargs):

        if lengths is None:
            lengths = range(0 if include_empty else 1, len(domain) + 1)

        for l in lengths:
            yield from itertools.permutations(domain, l)

    @op_def
    def Product(self, domain: Any, context: Any = None, **kwargs):
        yield from itertools.product(*domain)

    @op_def
    def Sequence(self, domain: Any, max_len: int = None, lengths: Collection[int] = None,
                  context: Any = None, **kwargs):
        if max_len is None and lengths is None:
            raise SyntaxError("Sequence requires the explicit keyword argument 'max_len' or 'lengths'")

        if max_len is not None and lengths is not None:
            raise SyntaxError("Sequence takes only *one* of the 'max_len' and 'lengths' keyword arguments")

        if max_len is not None:
            for l in range(1, max_len + 1):
                yield from itertools.product(domain, repeat=l)

        elif lengths is not None:
            for l in list(lengths):
                yield from itertools.product(domain, repeat=l)
