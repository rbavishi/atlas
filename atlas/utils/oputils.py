from collections import namedtuple
from typing import Optional, List


def create_sid(gen_name: str, gen_group: Optional[str],
               op_type: str, label: Optional[str], index: int):
    return f"{gen_group or ''}/{gen_name}/{op_type}@{label or ''}@{index}"


UnpackedSID = namedtuple("UnpackedSID", ['gen_group', 'gen_name', 'op_type', 'label', 'index'],
                         module='atlas.utils.oputils')


def unpack_sid(sid: str) -> UnpackedSID:
    gen_group, gen_name, base = sid.split('/')
    op_type, label, index = base.split('@')
    return UnpackedSID(
        gen_group=gen_group or None,
        gen_name=gen_name,
        op_type=op_type,
        label=label or None,
        index=int(index)
    )


class DefaultOpMethodResolver:
    def get_op_handler(self, sid: str, labels: Optional[List[str]]):
        op_type = unpack_sid(sid).op_type
        op_mro = []
        if labels is not None and len(labels) >= 1:
            op_mro.append(f"{op_type}_{labels[0]}")

        op_mro.append(op_type)
        for o in op_mro:
            if hasattr(self, o):
                return getattr(self, o)

        return None
