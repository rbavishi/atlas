import random
from typing import Callable, Optional, Collection, Any

from atlas import Strategy
from atlas.operators import OpInfo, operator


class RandStrategy(Strategy):
    def generic_op(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                   **kwargs):
        return handler(self, domain=domain, context=context, op_info=op_info, **kwargs)

    def is_finished(self):
        return False

    @operator
    def Select(self, domain: Any, **kwargs):
        return random.choice(domain)

    @operator
    def SelectFixed(self, domain: Any, **kwargs):
        return self.Select(domain)

    def Subset(self, domain: Any, context: Any = None, lengths: Collection[int] = None,
               include_empty: bool = False, **kwargs):
        if lengths is None:
            lengths = range(0 if include_empty else 1, len(domain) + 1)

        return random.sample(domain, random.choice(lengths))

    @operator
    def OrderedSubset(self, domain: Any, context: Any = None,
                      lengths: Collection[int] = None, include_empty: bool = False, **kwargs):
        if lengths is None:
            lengths = range(0 if include_empty else 1, len(domain) + 1)

        return random.sample(domain, random.choice(lengths))

    @operator
    def Sequence(self, domain: Any, context: Any = None, max_len: int = None,
                 lengths: Collection[int] = None, **kwargs):
        if max_len is None and lengths is None:
            raise SyntaxError("Sequence requires the explicit keyword argument 'max_len' or 'lengths'")

        if max_len is not None and lengths is not None:
            raise SyntaxError("Sequence takes only *one* of the 'max_len' and 'lengths' keyword arguments")

        if lengths is None:
            lengths = range(1, max_len + 1)

        return random.choices(domain, k=random.choice(lengths))

    @operator
    def SequenceFixed(self, domain: Any, context: Any = None, max_len: int = None,
                      lengths: Collection[int] = None, **kwargs):
        return self.Sequence(domain, context, max_len, lengths, **kwargs)

