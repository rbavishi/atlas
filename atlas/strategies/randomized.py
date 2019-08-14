import random
from typing import Callable, Optional, List

from atlas.strategies import Strategy, operator


class RandStrategy(Strategy):
    def generic_call(self, domain, context=None, sid: str = '',
                     labels: Optional[List[str]] = None, handler: Optional[Callable] = None, **kwargs):
        return handler(domain, context, sid, labels)

    def is_finished(self):
        return False

    @operator
    def Select(self, domain, *args, **kwargs):
        return random.choice(domain)

