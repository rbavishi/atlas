import textwrap
from typing import List, Optional

import atlas
from atlas.hooks import Hook


class OpTrace:
    def __init__(self, sid: str, choice, domain, context, **kwargs):
        self.sid = sid
        self.choice = choice
        self.domain = domain
        self.context = context

        self.kwargs = kwargs

    def __repr__(self):
        return textwrap.dedent(f"""
        OpTrace(sid={self.sid!r},
                choice={self.choice!r},
                domain={self.domain!r},
                context={self.context!r},
                **{self.kwargs!r}
               )""")


class GeneratorTrace:
    def __init__(self, f_inputs=None):
        self.f_inputs = f_inputs
        self.op_traces: List[OpTrace] = []

    def record_op_trace(self, op_trace: OpTrace):
        self.op_traces.append(op_trace)

    def __repr__(self):
        return repr(self.op_traces)


class DefaultTracer(Hook):
    def __init__(self):
        self.cur_trace: Optional[GeneratorTrace] = None

    def init_run(self, f_args, f_kwargs, **kwargs):
        self.cur_trace = GeneratorTrace((f_args, f_kwargs))

    def after_op(self, domain, context=None, retval=None, generator: 'Generator' = None,
                 sid: str = None, **kwargs):
        op_trace = OpTrace(sid=sid, choice=retval, domain=domain, context=context)
        self.cur_trace.record_op_trace(op_trace)

    def get_last_trace(self):
        return self.cur_trace
