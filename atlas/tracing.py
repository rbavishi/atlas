from typing import List, Optional

from atlas.hooks import Hook


class OpTrace:
    def __init__(self, choice, domain, context, **kwargs):
        self.choice = choice
        self.domain = domain
        self.context = context

        self.kwargs = kwargs


class GeneratorTrace:
    def __init__(self, f_inputs=None):
        self.f_inputs = f_inputs
        self.op_trace: List[OpTrace] = []

    def record_op_trace(self, op_trace: OpTrace):
        self.op_trace.append(op_trace)


class DefaultTracer(Hook):
    def __init__(self):
        self.cur_trace: Optional[GeneratorTrace] = None

    def init_run(self, f_args, f_kwargs, **kwargs):
        self.cur_trace = GeneratorTrace((f_args, f_kwargs))

    def after_op(self, domain, context=None, retval=None, op_name: str = None, sid: str = None, **kwargs):
        op_trace = OpTrace(retval, domain, context)
        self.cur_trace.record_op_trace(op_trace)

    def get_last_trace(self):
        return self.cur_trace
