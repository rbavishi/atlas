import itertools
from typing import Dict, Any, Callable, Collection, Optional, Iterator, Tuple

from atlas.exceptions import ExceptionAsContinue
from atlas.strategies import operator
from atlas.strategies.strategy import IteratorBasedStrategy
from atlas.operators import OpInfo


class DfsStrategy(IteratorBasedStrategy):
    def __init__(self, operator_iterator_bound: Optional[int] = None):
        super().__init__()
        self.call_id: int = 0
        self.op_iter_map: Dict[int, Iterator] = {}
        self.val_map: Dict[int, Any] = {}
        self.finished: bool = False

        #  This optimization is semantically correct if and only if the generator is deterministic modulo
        #  operator choices and side-effect free i.e. the generator follows the same execution path and
        #  returns the same result if all the operators make the same choices and does not mutate any object

        #  The cache contains tuples as values with the first two elements being the start and end call-id,
        #  and the third being the value returned by the generator
        self.gen_call_id = 0
        self.gen_result_cache: Dict[int, Tuple[int, int, Any]] = {}

        #  Bound on number of values to return for each operator
        self.operator_iterator_bound = operator_iterator_bound

    def init(self):
        self.call_id = 0
        self.gen_call_id = 0
        self.op_iter_map = {}
        self.gen_result_cache = {}
        self.finished = False

    def init_run(self):
        self.call_id = 0
        self.gen_call_id = 0

    def finish_run(self):
        for t in range(max(self.op_iter_map.keys(), default=-1), -1, -1):
            try:
                self.val_map[t] = next(self.op_iter_map[t])
                self.val_map = {k: v for k, v in self.val_map.items() if k <= t}
                self.op_iter_map = {k: v for k, v in self.op_iter_map.items() if k <= t}

                #  The call id of the last operator in the generator <= t for it to be cached correctly
                self.gen_result_cache = {k: v for k, v in self.gen_result_cache.items() if 0 <= v[1] <= t}
                return

            except StopIteration:
                continue

        self.finished = True

    def is_finished(self):
        return self.finished

    def generator_invoked(self):
        k = self.gen_call_id
        self.gen_call_id += 1
        self.gen_result_cache[k] = (self.call_id, -1, None)
        return k

    def generator_returned(self, gen_call_id: int, result: Any):
        start, _, _ = self.gen_result_cache[gen_call_id]
        self.gen_result_cache[gen_call_id] = (start, self.call_id, result)

    def cached_generator_invocation(self):
        if self.gen_call_id in self.gen_result_cache:
            entry = self.gen_result_cache[self.gen_call_id]
            assert entry[0] == self.call_id
            self.gen_call_id += 1
            self.call_id = entry[1]
            return True, entry[2]

        return False, False

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

                if self.operator_iterator_bound:
                    op_iter = itertools.islice(iter(iterator), self.operator_iterator_bound)
                else:
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
    def Select(self, domain: Any, context: Any = None, **kwargs):
        yield from domain

    @operator
    def SelectFixed(self, domain: Any, context: Any = None, **kwargs):
        yield from self.Select(domain, context, **kwargs)

    @operator
    def Substr(self, domain: Any, context: Any = None, **kwargs):
        if isinstance(domain, str):
            for i in range(len(domain)):
                for j in range(i, len(domain)):
                    yield domain[i: j+1]

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

    @operator
    def SequenceFixed(self, domain: Any, context: Any = None, max_len: int = None,
                      lengths: Collection[int] = None, **kwargs):
        return self.Sequence(domain, context, max_len, lengths, **kwargs)

