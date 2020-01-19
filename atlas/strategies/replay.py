import collections
from typing import Callable, Optional, List, Union, Dict, Iterator

from atlas import Strategy
from atlas.operators import OpInfo, operator
from atlas.tracing import GeneratorTrace


class FullReplayStrategy(Strategy):
    """
    Replay a full GeneratorTrace. Throws an error if the trace and the generator execution
    are inconsistent at any point of time. This restriction is not imposed by the PartialReplayStrategy below.
    FullReplayStrategy is used when `replay` is called on a `Generator` object.
    The base strategy is used to add a basis for the operators in order to compile the generator.
    However the strategy itself will never be called.
    """
    def __init__(self, trace: GeneratorTrace, base_strategy: Strategy):
        super().__init__()
        self.trace = trace
        self.known_ops = base_strategy.known_ops
        self.op_choices = collections.defaultdict(list)
        for t in trace.op_traces:
            self.op_choices[t.op_info.sid].append(t.choice)

        self.op_choice_iter_map: Dict[str, Iterator] = {}

    def is_finished(self):
        return False

    def init_run(self):
        self.op_choice_iter_map = {k: iter(v) for k, v in self.op_choices.items()}

    def generic_op(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                   *args, **kwargs):

        if op_info.sid in self.op_choice_iter_map:
            return next(self.op_choice_iter_map[op_info.sid])

        raise KeyError(f"Generator and trace are inconsistent. "
                       f"Choice could not be made for operator with sid {op_info.sid}")


class PartialReplayStrategy(Strategy):
    """
    Replay a GeneratorTrace or a Mapping from sid/uids to return values of operators.
    It also takes a backup strategy as an argument to consult if an operator is encountered
    for which no replay information is available. Consequently, it does not throws an error
    if the trace and the generator execution are inconsistent at any point of time.
    PartialReplayStrategy is used when `with_replay` is called on a `GeneratorExecEnvironment` object.
    """
    def __init__(self, trace: Union[Dict[str, List], GeneratorTrace], backup_strategy: Strategy):
        super().__init__()
        self.trace = trace
        self.backup_strategy = backup_strategy
        self.known_ops = backup_strategy.known_ops

        self.op_choices = collections.defaultdict(list)
        self.uid_choices = {}

        if isinstance(trace, GeneratorTrace):
            for t in trace.op_traces:
                self.op_choices[t.op_info.sid].append(t.choice)

        else:
            self.uid_choices = trace.copy()

        self.op_choice_map: Dict[str, Iterator] = {}
        self.uid_choice_map: Dict[str, Iterator] = {}

    def get_op_handler(self, op_info: OpInfo):
        return self.backup_strategy.get_op_handler(op_info)

    def is_finished(self):
        return self.backup_strategy.is_finished()

    def init(self):
        self.backup_strategy.init()

    def init_run(self):
        self.op_choice_map = {k: iter(v) for k, v in self.op_choices.items()}
        self.uid_choice_map = {k: iter(v) for k, v in self.uid_choices.items()}
        self.backup_strategy.init_run()

    def finish_run(self):
        self.backup_strategy.finish_run()

    def finish(self):
        self.backup_strategy.finish()

    def generic_op(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                   *args, **kwargs):
        if op_info.sid in self.op_choice_map:
            return next(self.op_choice_map[op_info.sid])

        if op_info.uid in self.uid_choice_map:
            return next(self.uid_choice_map[op_info.uid])

        return self.backup_strategy.generic_op(domain, context=context, op_info=op_info,
                                               handler=handler, **kwargs)

    def __getattr__(self, item):
        return self.backup_strategy.__getattribute__(item)
