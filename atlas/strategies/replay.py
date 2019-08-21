import collections
from typing import Callable, Optional, List

from atlas.operators import OpInfo
from atlas.strategies import Strategy, operator
from atlas.tracing import GeneratorTrace


class ReplayStrategy(Strategy):
    def __init__(self, trace: GeneratorTrace, backup_strategy: Strategy):
        super().__init__()
        self.trace = trace
        self.backup_strategy = backup_strategy
        self.known_ops = backup_strategy.known_ops

        self.op_choice_map = collections.defaultdict(list)
        for t in trace.op_traces:
            self.op_choice_map[t.op_info.sid].append(t.choice)

        self.op_choice_map = {k: iter(v) for k, v in self.op_choice_map.items()}

    def get_op_handler(self, op_info: OpInfo):
        return self.backup_strategy.get_op_handler(op_info)

    def is_finished(self):
        return self.backup_strategy.is_finished()

    def init(self):
        self.backup_strategy.init()

    def init_run(self):
        self.backup_strategy.init_run()

    def finish_run(self):
        self.backup_strategy.finish_run()

    def finish(self):
        self.backup_strategy.finish()

    def generic_call(self, domain, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     *args, **kwargs):
        if op_info.sid in self.op_choice_map:
            return next(self.op_choice_map[op_info.sid])

        return self.backup_strategy.generic_call(domain, context=context, op_info=op_info,
                                                 handler=handler, *args, **kwargs)
