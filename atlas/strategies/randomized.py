import random
from typing import Callable, Optional

from atlas.strategies import Strategy, operator
from atlas.operators import OpInfo


class RandStrategy(Strategy):
    def generic_call(self, domain, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     *args, **kwargs):
        return handler(domain, context, op_info)

    def is_finished(self):
        return False

    @operator
    def Select(self, domain, *args, **kwargs):
        return random.choice(domain)

