from typing import Callable, Any

from atlas.semantics import op_def
from atlas.semantics.base import PyGeneratorBasedSemantics


class DfsExperimental(PyGeneratorBasedSemantics):
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
