from typing import Optional


class OpEncoder:
    def get_encoder(self, op_name: str, sid: str, oid: Optional[str]):
        if oid is not None:
            return getattr(self, op_name + "_" + oid,
                           getattr(self, op_name))

        return getattr(self, op_name)


