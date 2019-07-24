import random
from typing import Callable

from atlas.strategies import Strategy, operator


class RandStrategy(Strategy):
    def make_op(self, op_kind: str, op_id: str) -> Callable:
        label = op_kind
        if op_kind + "_" + op_id in dir(self):
            label = op_kind + "_" + op_id

        return getattr(self, label)

    def is_finished(self):
        return False

    @operator
    def Select(self, domain, *args, **kwargs):
        return random.choice(domain)

