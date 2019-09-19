import ast
import collections
from typing import Optional, List, NamedTuple, Callable, Union

import astunparse

IS_GENERATOR_OP = "_is_generator_operator"
IS_GENERATOR_METHOD = "_is_generator_method"
RESOLUTION_INFO = "_resolution_INFO"


def operator_decorator(name: str = None,
                       uid: str = None,
                       tags: Union[str, List[str]] = None,
                       gen_name: str = None,
                       gen_group: str = None):
    def wrapper(func: Callable):
        setattr(func, IS_GENERATOR_OP, True)
        setattr(func, RESOLUTION_INFO, {'name': name or func.__name__, 'uid': uid, 'tags': tags,
                                        'gen_name': gen_name, 'gen_group': gen_group})
        return func

    return wrapper


def operator(*args, **kwargs) -> Callable:
    """
    Can be used with no arguments or specific keyword arguments to define an operator
    inside a strategy as follows -

    .. code-block:: python

        from atlas.strategies import DfsStrategy, operator

        class TestStrategy(DfsStrategy):
            @operator
            def Select(*args, **kwargs):
                #  Code for calls to Select by default
                pass

            @operator(name='Select', uid="10")
            def CustomSelectForUid10(*args, **kwargs):
                #  Custom code for the particular call to Select with uid=10
                pass


    The function also accepts specific keyword arguments:

    Keyword Args:
        name (str): Name of the operator to override (required)

        uid (str): UID of the operator to override (matches all UIDs by default)

        tags (Union[str, List[str]]): Tags of the operator to match (matches all by default)

        gen_name (str): Name of the generator inside which to match the operator (matches all generators by default)

        gen_group (str): Name of the generator group whose generators need to be peeked inside to match the operator.
            Matches all by default.
    """

    allowed_kwargs = {'name', 'uid', 'tags', 'gen_name', 'gen_group'}
    error_str = f"The @operator decorator should be applied either with no parentheses or " \
                f"at least one of the following keyword args - {', '.join(allowed_kwargs)}."
    assert (len(args) == 1 and len(kwargs) == 0 and callable(args[0])) or \
           (len(args) == 0 and len(kwargs) > 0 and set(kwargs.keys()).issubset(allowed_kwargs)), error_str

    if len(args) == 1:
        return operator_decorator()(args[0])

    else:
        return operator_decorator(**kwargs)


def method(func: Callable):
    setattr(func, IS_GENERATOR_METHOD, True)
    return func


def is_operator(func):
    return getattr(func, IS_GENERATOR_OP, False)


def is_method(func):
    return getattr(func, IS_GENERATOR_METHOD, False)


def resolve(func):
    return getattr(func, RESOLUTION_INFO)


class OpInfo(NamedTuple):
    sid: str
    gen_name: str
    op_type: str
    index: int
    gen_group: str = None
    uid: Optional[str] = None
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

    def extract_uid(self, op_call: ast.Call) -> Optional[str]:
        uid = self.find_and_remove_keyword(op_call, 'uid')
        if uid is None:
            return None

        if not isinstance(uid, ast.Str):
            raise SyntaxError(f"Label passed to operator must be a string constantin {astunparse.unparse(op_call)}")

        return uid.s

    def extract_tags(self, op_call: ast.Call) -> Optional[List[str]]:
        tags = self.find_and_remove_keyword(op_call, 'tags')
        if tags is None:
            return None

        if (not isinstance(tags, (ast.List, ast.Tuple))) or (not all(isinstance(i, ast.Str) for i in tags.elts)):
            raise SyntaxError(f"Tags passed to operator must be a list/tuple of "
                              f"string constants in {astunparse.unparse(op_call)}")

        return [i.s for i in tags.elts]

    def get(self, op_call: ast.Call, gen_name: str, gen_group: Optional[str]) -> OpInfo:
        uid: Optional[str] = self.extract_uid(op_call)
        tags: Optional[List[str]] = self.extract_tags(op_call)
        op_type: str = op_call.func.id

        sid_key = (gen_name, gen_group, op_type, uid)
        self.sid_index_map[sid_key] += 1
        index = self.sid_index_map[sid_key]
        sid = create_sid(gen_name, gen_group, op_type, uid, index)

        return OpInfo(
            sid=sid,
            gen_name=gen_name,
            op_type=op_type,
            index=index,
            uid=uid,
            tags=tags,
            gen_group=gen_group
        )


def create_sid(gen_name: str, gen_group: Optional[str],
               op_type: str, uid: Optional[str], index: int):
    return f"{gen_group or ''}/{gen_name}/{op_type}@{uid or ''}@{index}"


class UnpackedSID(NamedTuple):
    gen_group: str
    gen_name: str
    op_type: str
    uid: str
    index: int


def unpack_sid(sid: str) -> UnpackedSID:
    gen_group, gen_name, base = sid.split('/')
    op_type, uid, index = base.split('@')
    return UnpackedSID(
        gen_group=gen_group or None,
        gen_name=gen_name,
        op_type=op_type,
        uid=uid or None,
        index=int(index)
    )
