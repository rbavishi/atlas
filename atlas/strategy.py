import collections
from abc import abstractmethod, ABC
from typing import Callable, Optional, Set, List

from atlas.exceptions import ExceptionAsContinue
from atlas.hooks import Hook
from atlas.models import GeneratorModel
from atlas.operators import OpInfo, is_operator, is_method, get_attrs, OpResolvable, resolve_operator, \
    find_known_operators, find_known_methods


class Strategy(ABC, OpResolvable):
    def __init__(self):
        self.known_ops = find_known_operators(self)
        self.known_methods = find_known_methods(self)

    def get_op_handler(self, op_info: OpInfo):
        return resolve_operator(self.known_ops, op_info)

    def init(self):
        pass

    def finish(self):
        pass

    def init_run(self):
        pass

    def finish_run(self):
        pass

    @abstractmethod
    def is_finished(self):
        pass

    def get_known_ops(self):
        return self.known_ops

    def get_known_methods(self):
        return self.known_methods

    def generic_op(self, domain=None, context=None, model: GeneratorModel = None,
                   op_info: OpInfo = None, handler: Optional[Callable] = None,
                   **kwargs):
        pass

    def gen_iterate(self, func: Callable, args, kwargs, atlas_kwargs, hooks: List[Hook], gen: 'Generator',
                    ignore_exceptions: bool = False):
        for h in hooks:
            h.init(args, kwargs)

        self.init()
        while not self.is_finished():
            for h in hooks:
                h.init_run(args, kwargs)

            self.init_run()
            try:
                yield func(*args, **kwargs, **atlas_kwargs)

            except ExceptionAsContinue:
                pass

            except Exception:
                if not ignore_exceptions:
                    raise

            self.finish_run()

            for h in hooks:
                h.finish_run()

        self.finish()

        for h in hooks:
            h.finish()

    def gen_call(self, func: Callable, args, kwargs, atlas_kwargs, gen: 'Generator'):
        return func(*args, **kwargs, **atlas_kwargs)
