import itertools
from typing import Dict, Any, Callable, Collection, Optional, List

from atlas.exceptions import ExceptionAsContinue
from atlas.strategies import operator
from atlas.strategies.strategy import IteratorBasedStrategy
from atlas.utils.iterutils import PeekableGenerator
from atlas.utils.oputils import OpInfo


class DfsStrategy(IteratorBasedStrategy):
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

    def generic_call(self, domain, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     *args, **kwargs):
        t = self.call_id
        self.call_id += 1

        if t not in self.op_map:
            try:
                iterator = None
                if self.model is not None:
                    try:
                        iterator = self.model.infer(domain, context=context, op_info=op_info, **kwargs)
                    except NotImplementedError:
                        pass

                if iterator is None:
                    iterator = handler(domain, context=context, op_info=op_info, **kwargs)

                op: PeekableGenerator = PeekableGenerator(iter(iterator))

            except StopIteration:
                #  Operator received an empty domain
                raise ExceptionAsContinue

            self.op_map[t] = op

        else:
            op = self.op_map[t]

        if not op.is_finished():
            self.last_unfinished = t

        return op.peek()

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
