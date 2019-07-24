import random
from typing import Callable

from atlas.strategies import Strategy, operator


class RandStrategy(Strategy):
    def make_op(self, kind: str, sid: str) -> Callable:
        label = kind
        if kind + "_" + sid in dir(self):
            label = kind + "_" + sid

        return getattr(self, label)

    def is_finished(self):
        return False

    @operator
    def Select(self, domain, *args, **kwargs):
        return random.choice(domain)

