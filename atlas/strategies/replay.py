from typing import Callable, Optional, List

from atlas.strategies import Strategy, operator
from atlas.tracing import GeneratorTrace


class Replay(Strategy):
    def __init__(self, traces: List[GeneratorTrace], repeat: bool = False):
        super().__init__()
        self.traces = traces
        self.op_choice_map = {}
        self._trace_num: int = 0
        self._cur_trace = None
        self.repeat = repeat
        self.set_known_ops()

    def set_known_ops(self):
        self.known_ops = {o.op_name for t in self.traces for o in t.op_traces}

    def is_finished(self):
        return (not self.repeat) and self._trace_num == len(self.traces) - 1

    def init(self):
        self._trace_num = -1

    def init_run(self):
        self._trace_num = (self._trace_num + 1) % len(self.traces)
        self._cur_trace: GeneratorTrace = self.traces[self._trace_num]
        self.op_choice_map = {t.sid: t.choice for t in self._cur_trace.op_traces}

    def make_op(self, gen: 'Generator', op_name: str, sid: str, oid: Optional[str]) -> Callable:
        def wrapper(*args, **kwargs):
            if sid not in self.op_choice_map:
                raise KeyError(f"Could not find Op with SID {sid} in trace {self._cur_trace}")

            return self.op_choice_map[sid]

        return wrapper
