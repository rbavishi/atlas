from abc import ABC


class Hook(ABC):
    def init(self, f_args, f_kwargs, **kwargs):
        pass

    def init_run(self):
        pass

    def finish_run(self):
        pass

    def finish(self):
        pass

    def before_op(self, domain, context=None, op_name: str = None, sid: str = None, **kwargs):
        pass

    def after_op(self, domain, context=None, retval=None, op_name: str = None, sid: str = None, **kwargs):
        pass
