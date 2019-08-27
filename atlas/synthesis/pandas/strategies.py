from atlas.operators import OpInfo
from atlas.strategies import DfsStrategy, operator


class PandasSynthesisStrategy(DfsStrategy):
    @operator
    def SelectInput(self, domain, dtype=None, context=None, op_info: OpInfo = None, **kwargs):
        yield from (i for i in domain if isinstance(i, dtype))
