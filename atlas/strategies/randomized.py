import random
from typing import Callable, Optional, Collection

from atlas.operators import OpInfo
from atlas.strategies import Strategy, operator


class RandStrategy(Strategy):
    def generic_call(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     **kwargs):
        return handler(domain=domain, context=context, op_info=op_info, **kwargs)

    def is_finished(self):
        return False

    @operator
    def Select(self, domain, **kwargs):
        return random.choice(domain)

    @operator
    def Subset(self, domain, lengths: Collection[int] = None, **kwargs):
        if lengths is None:
            lengths = list(range(1, len(domain) + 1))

        return random.sample(domain, random.choice(lengths))
