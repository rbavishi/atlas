from abc import ABC
from typing import Any

import atlas


class Hook(ABC):
    def init(self, f_args, f_kwargs, **kwargs):
        pass

    def init_run(self, f_args, f_kwargs, **kwargs):
        pass

    def finish_run(self):
        pass

    def finish(self):
        pass

    def before_op(self, domain, context=None, sid: str = None, **kwargs):
        pass

    def after_op(self, domain, context=None, sid: str = None, retval: Any = None, **kwargs):
        pass
