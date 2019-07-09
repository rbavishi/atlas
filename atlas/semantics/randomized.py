import random
from typing import Callable

from atlas.semantics import Semantics, op_def


class RandSemantics(Semantics):
    def make_call(self, op_kind: str, op_id: str) -> Callable:
        label = op_kind
        if op_kind + "_" + op_id in dir(self):
            label = op_kind + "_" + op_id

        return getattr(self, label)

    def is_finished(self):
        return False

    @op_def
    def Select(self, domain, *args, **kwargs):
        return random.choice(domain)

