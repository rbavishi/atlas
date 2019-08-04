import textwrap
from typing import List, Optional, Any

import atlas
from atlas.hooks import Hook
from atlas.utils.genutils import unpack_sid


class OpTrace:
    def __init__(self, choice, domain, context, sid: str, op_type: str, oid: Optional[str],
                 gen_group: Optional[str], gen_name: str, index: int, **kwargs):
        self.choice = choice
        self.domain = domain
        self.context = context
        self.sid = sid
        self.op_type = op_type
        self.oid = oid
        self.gen_group = gen_group
        self.gen_name = gen_name
        self.index = index

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

    def after_op(self, domain, context=None, sid: str = None, retval: Any = None, **kwargs):
        unpacked = unpack_sid(sid)
        gen_group = unpacked.gen_group
        gen_name = unpacked.gen_name
        op_type = unpacked.op_type
        oid = unpacked.oid
        index = unpacked.index
        op_trace = OpTrace(sid=sid, choice=retval, domain=domain, context=context, op_type=op_type, oid=oid,
                           gen_group=gen_group, gen_name=gen_name, index=index)
        self.cur_trace.record_op_trace(op_trace)

    def get_last_trace(self):
        return self.cur_trace
