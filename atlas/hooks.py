from abc import ABC

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

    def before_op(self, domain, context=None, generator: 'Generator' = None,
                  sid: str = None, **kwargs):
        pass

    def after_op(self, domain, context=None, retval=None, generator: 'Generator' = None,
                 sid: str = None, **kwargs):
        pass
