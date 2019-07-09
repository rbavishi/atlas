from typing import Dict, Any, Callable, Generator

from atlas.semantics import Semantics, op_def


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


class DfsSemantics(Semantics):
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

        def call(domain: Any, context: Any = None, **kwargs):
            t = self.call_id
            self.call_id += 1

            if t not in self.op_map:
                op: PeekableGenerator = PeekableGenerator(handler(domain, context, op_id=op_id))
                self.op_map[t] = op

            else:
                op = self.op_map[t]

            if not op.is_finished():
                self.last_unfinished = t

            return op.peek()

        return call

    @op_def
    def Select(self, domain: Any, context: Any = None, **kwargs):
        yield from domain
