import ast
import collections

import astunparse
from collections import namedtuple
from typing import Optional, List, NamedTuple


class OpInfo(NamedTuple):
    sid: str
    gen_name: str
    op_type: str
    index: int
    gen_group: str = None
    label: Optional[str] = None
    tags: Optional[List[str]] = None


class OpInfoConstructor:
    def __init__(self):
        self.sid_index_map = collections.defaultdict(int)

    def find_and_remove_keyword(self, op_call: ast.Call, arg: str) -> Optional[ast.AST]:
        for idx, kw in enumerate(op_call.keywords):
            if kw.arg == arg:
                op_call.keywords.pop(idx)
                return kw.value

        return None

    def extract_label(self, op_call: ast.Call) -> Optional[str]:
        label = self.find_and_remove_keyword(op_call, 'label')
        if label is None:
            return None

        if not isinstance(label, ast.Str):
            raise SyntaxError(f"Label passed to operator must be a string constantin {astunparse.unparse(op_call)}")

        return label.s

    def extract_tags(self, op_call: ast.Call) -> Optional[List[str]]:
        tags = self.find_and_remove_keyword(op_call, 'tags')
        if tags is None:
            return None

        if (not isinstance(tags, (ast.List, ast.Tuple))) or (not all(isinstance(i, ast.Str) for i in tags.elts)) :
            raise SyntaxError(f"Tags passed to operator must be a list/tuple of "
                              f"string constants in {astunparse.unparse(op_call)}")

        return [i.s for i in tags.elts]

    def get(self, op_call: ast.Call, gen_name: str, gen_group: Optional[str]) -> OpInfo:
        label: Optional[str] = self.extract_label(op_call)
        tags: Optional[List[str]] = self.extract_tags(op_call)
        op_type: str = op_call.func.id

        sid_key = (gen_name, gen_group, op_type, label)
        self.sid_index_map[sid_key] += 1
        index = self.sid_index_map[sid_key]
        sid = create_sid(gen_name, gen_group, op_type, label, index)

        return OpInfo(
            sid=sid,
            gen_name=gen_name,
            op_type=op_type,
            index=index,
            label=label,
            tags=tags,
            gen_group=gen_group
        )


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
    def get_op_handler(self, op_info: OpInfo):
        label = op_info.label
        op_type = op_info.op_type

        op_mro = []
        if label is not None:
            op_mro.append(f"{op_type}_{label}")

        op_mro.append(op_type)
        for o in op_mro:
            if hasattr(self, o):
                return getattr(self, o)

        return None
