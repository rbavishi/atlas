from typing import Callable, Any

from atlas.strategies import op_def
from atlas.strategies.base import PyGeneratorBasedStrategy


class DfsExperimental(PyGeneratorBasedStrategy):
    def make_call(self, op_kind: str, op_id: str) -> Callable:
        label = op_kind
        if op_kind + "_" + op_id in dir(self):
            label = op_kind + "_" + op_id

        return getattr(self, label)

    def is_finished(self):
        return False

    @op_def
    def Select(self, domain: Any, context: Any = None, **kwargs):
        yield from domain
