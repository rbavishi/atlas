import random
from typing import Callable, Optional, Union

from atlas.strategies import Strategy, operator
from atlas.operators import OpInfo


class RandStrategy(Strategy):
    def generic_call(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     **kwargs):
        return handler(domain=domain, context=context, op_info=op_info, **kwargs)

    def is_finished(self):
        return False

    @operator
    def Select(self, domain, **kwargs):
        return random.choice(domain)

