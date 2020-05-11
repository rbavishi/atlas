import textwrap
from typing import List, Optional, Any

from atlas.hooks import Hook
from atlas.operators import OpInfo


class OpTrace:
    def __init__(self, choice, domain, context, op_info: OpInfo, **kwargs):
        self.choice = choice
        self.domain = domain
        self.context = context
        self.op_info = op_info

        self.kwargs = kwargs

    def __repr__(self):
        return textwrap.dedent(f"""
        OpTrace(op_info={self.op_info!r},
                choice={self.choice!r},
                domain={self.domain!r},
                context={self.context!r},
                **{self.kwargs!r}
               )""")

    def copy(self):
        return OpTrace(
            choice=self.choice,
            domain=self.domain,
            context=self.context,
            op_info=self.op_info,
            **self.kwargs
        )


class GeneratorTrace:
    def __init__(self, f_inputs=None):
        self.f_inputs = f_inputs
        self.op_traces: List[OpTrace] = []

    def record_op_trace(self, op_trace: OpTrace):
        self.op_traces.append(op_trace)

    def __repr__(self):
        return textwrap.dedent(f"""
        GeneratorTrace(inputs={self.f_inputs},
                       op_traces={self.op_traces!r}
        """)

    def copy(self):
        r = GeneratorTrace()
        r.f_inputs = self.f_inputs[:]
        r.op_traces = [o.copy() for o in self.op_traces]

        return r


class DefaultTracer(Hook):
    def __init__(self):
        self.cur_trace: Optional[GeneratorTrace] = None

    def init_run(self, f_args, f_kwargs, **kwargs):
        self.cur_trace = GeneratorTrace((f_args, f_kwargs))

    def after_op(self, domain=None, context=None, op_info: OpInfo = None, retval: Any = None, **kwargs):
        op_trace = OpTrace(op_info=op_info, choice=retval, domain=domain, context=context, **kwargs)
        self.cur_trace.record_op_trace(op_trace)

    def get_last_trace(self):
        return self.cur_trace
