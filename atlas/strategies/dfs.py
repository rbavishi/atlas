import itertools
from typing import Dict, Any, Callable, Generator, Collection, Optional

from atlas.exceptions import ExceptionAsContinue
from atlas.strategies import Strategy, operator


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

    def make_op(self, op_name: str, sid: str, oid: Optional[str]) -> Callable:
        label = op_name
        if oid is not None and op_name + "_" + oid in dir(self):
            label = op_name + "_" + oid

        handler = getattr(self, label)

        def call(domain: Any, context: Any = None, **kwargs):
            t = self.call_id
            self.call_id += 1

            if t not in self.op_map:
                try:
                    op: PeekableGenerator = PeekableGenerator(handler(domain, context=context, sid=sid, **kwargs))

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

    @operator
    def Select(self, domain: Any, context: Any = None, fixed_domain=False, **kwargs):
        yield from domain

    @operator
    def Subset(self, domain: Any, context: Any = None, lengths: Collection[int] = None,
               include_empty: bool = False, **kwargs):
        if lengths is None:
            lengths = range(0 if include_empty else 1, len(domain) + 1)

        for l in lengths:
            yield from itertools.combinations(domain, l)

    @operator
    def OrderedSubset(self, domain: Any, context: Any = None,
                      lengths: Collection[int] = None, include_empty: bool = False, **kwargs):

        if lengths is None:
            lengths = range(0 if include_empty else 1, len(domain) + 1)

        for l in lengths:
            yield from itertools.permutations(domain, l)

    @operator
    def Product(self, domain: Any, context: Any = None, **kwargs):
        yield from itertools.product(*domain)

    @operator
    def Sequence(self, domain: Any, context: Any = None, max_len: int = None,
                 lengths: Collection[int] = None, **kwargs):
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
