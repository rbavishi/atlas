from abc import ABC
from typing import Optional


class OpEncoder(ABC):
    def get_encoder(self, op_name: str, sid: str, oid: Optional[str]):
        if oid is not None:
            return getattr(self, op_name + "_" + oid,
                           getattr(self, op_name))

        return getattr(self, op_name)


class ParallelizedEncoder(OpEncoder, ABC):
    pass


class StatefulEncoder(OpEncoder, ABC):
    pass


