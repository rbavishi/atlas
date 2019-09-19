import collections
from abc import abstractmethod, ABC
from typing import Callable, Optional, Set

from atlas.models.core import GeneratorModel
from atlas.operators import OpInfo, is_operator, is_method, resolve


class Strategy(ABC):
    def __init__(self):
        self.known_ops = collections.defaultdict(list)
        self.known_methods = set()
        self.collect_ops_and_methods()

    def collect_ops_and_methods(self):
        for k in dir(self):
            v = getattr(self, k)
            if is_operator(v):
                attrs = resolve(v)
                self.known_ops[attrs['name']].append((getattr(type(self), k), attrs))

            if is_method(v):
                self.known_methods.add(k)

    def get_op_handler(self, op_info: OpInfo):
        handlers = self.known_ops[op_info.op_type]
        if len(handlers) == 1:
            return handlers[0][0]

        #  First filter out downright mismatches
        handlers = [h for h in handlers if h[1]['gen_name'] in [None, op_info.gen_name]]
        handlers = [h for h in handlers if h[1]['gen_group'] in [None, op_info.gen_group]]
        handlers = [h for h in handlers if h[1]['uid'] in [None, op_info.uid]]
        handlers = [h for h in handlers if set(h[1]['tags'] or op_info.tags or []).issuperset(set(op_info.tags or []))]

        #  Get the "most-specific" matches i.e. handlers with the most number of fields specified (not None)
        min_none_cnts = min(list(h[1].values()).count(None) for h in handlers)
        handlers = [h for h in handlers if list(h[1].values()).count(None) == min_none_cnts]

        if len(handlers) == 1:
            return handlers[0][0]

        raise ValueError(f"Could not resolve operator handler unambiguously for operator {op_info}.")

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

    def generic_call(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     **kwargs):
        pass


class IteratorBasedStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()
        self.model: Optional[GeneratorModel] = None

    def set_model(self, model: Optional[GeneratorModel]):
        self.model = model
