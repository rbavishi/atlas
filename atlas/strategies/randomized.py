import random
from typing import Callable, Optional

from atlas.strategies import Strategy, operator


class RandStrategy(Strategy):
    def make_op(self, op_name: str, sid: str, oid: Optional[str]) -> Callable:
        label = op_name
        if oid is not None and op_name + "_" + oid in dir(self):
            label = op_name + "_" + oid

        return getattr(self, label)

    def is_finished(self):
        return False

    @operator
    def Select(self, domain, *args, **kwargs):
        return random.choice(domain)

