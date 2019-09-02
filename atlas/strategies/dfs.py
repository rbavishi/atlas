import itertools
from typing import Dict, Any, Callable, Collection, Optional, Iterator

from atlas.exceptions import ExceptionAsContinue
from atlas.strategies import operator
from atlas.strategies.strategy import IteratorBasedStrategy
from atlas.operators import OpInfo


class DfsStrategy(IteratorBasedStrategy):
    def __init__(self):
        super().__init__()
        self.call_id: int = 0
        # self.op_iter_map: Dict[int, PeekableGenerator] = {}
        self.op_iter_map: Dict[int, Iterator] = {}
        self.val_map: Dict[int, Any] = {}
        self.finished: bool = False

    def init(self):
        self.call_id = 0
        self.op_iter_map = {}
        self.finished = False

    def init_run(self):
        self.call_id = 0

    def finish_run(self):
        for t in range(self.call_id - 1, -1, -1):
            try:
                self.val_map[t] = next(self.op_iter_map[t])
                self.val_map = {k: v for k, v in self.val_map.items() if k <= t}
                self.op_iter_map = {k: v for k, v in self.op_iter_map.items() if k <= t}
                return

            except StopIteration:
                continue

        self.finished = True

    def is_finished(self):
        return self.finished

    def generic_call(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     **kwargs):
        t = self.call_id
        self.call_id += 1

        if t not in self.op_iter_map:
            try:
                iterator = None
                if self.model is not None:
                    try:
                        iterator = self.model.infer(domain=domain, context=context, op_info=op_info, **kwargs)
                    except NotImplementedError:
                        pass

                if iterator is None:
                    iterator = handler(self, domain=domain, context=context, op_info=op_info, **kwargs)

                op_iter = iter(iterator)
                self.op_iter_map[t] = op_iter
                val = self.val_map[t] = next(op_iter)

            except StopIteration:
                #  Operator received an empty domain
                raise ExceptionAsContinue

        else:
            val = self.val_map[t]

        return val

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
