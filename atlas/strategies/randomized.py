import random
from typing import Callable

from atlas.strategies import Strategy, operator


class RandStrategy(Strategy):
    def make_op(self, op_name: str, sid: str) -> Callable:
        label = op_name
        if op_name + "_" + sid in dir(self):
            label = op_name + "_" + sid

        return getattr(self, label)

    def is_finished(self):
        return False

    @operator
    def Select(self, domain, *args, **kwargs):
        return random.choice(domain)

