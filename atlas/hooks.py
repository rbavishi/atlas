from abc import ABC
from typing import Any

from atlas.operators import OpInfo


class Hook(ABC):
    def init(self, f_args, f_kwargs, **kwargs):
        pass

    def init_run(self, f_args, f_kwargs, **kwargs):
        pass

    def finish_run(self):
        pass

    def finish(self):
        pass

    def before_op(self, domain=None, context=None, op_info: OpInfo = None, **kwargs):
        pass

    def after_op(self, domain=None, context=None, op_info: OpInfo = None, retval: Any = None, **kwargs):
        pass
