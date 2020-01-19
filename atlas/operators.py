import ast
import collections
from typing import Optional, List, NamedTuple, Callable, Union, Dict, Tuple

import astunparse

IS_GENERATOR_OP = "_is_generator_operator"
IS_GENERATOR_METHOD = "_is_generator_method"
RESOLUTION_INFO = "_resolution_INFO"
RETURNS_LAMBDA = "_returns_lambda"


class OpInfo(NamedTuple):
    sid: str
    gen_name: str
    op_type: str
    index: int
    gen_group: str = None
    uid: Optional[str] = None
    tags: Optional[Tuple[str, ...]] = None


def returns_lambda(handler):
    return getattr(handler, RETURNS_LAMBDA, False)


def operator_decorator(name: str = None,
                       uid: str = None,
                       tags: Union[str, List[str]] = None,
                       gen_name: str = None,
                       gen_group: str = None,
                       returns_lambda: bool = False):
    def wrapper(func: Callable):
        setattr(func, IS_GENERATOR_OP, True)
        setattr(func, RESOLUTION_INFO, {'name': name or func.__name__, 'uid': uid, 'tags': tags,
                                        'gen_name': gen_name, 'gen_group': gen_group})
        setattr(func, RETURNS_LAMBDA, returns_lambda)
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

    allowed_kwargs = {'name', 'uid', 'tags', 'gen_name', 'gen_group', 'returns_lambda'}
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


def get_attrs(func):
    return getattr(func, RESOLUTION_INFO)


class OpResolvable:
    pass


def find_known_operators(obj: OpResolvable):
    known_ops = collections.defaultdict(list)
    for k in dir(obj):
        v = getattr(obj, k)
        if is_operator(v):
            attrs = get_attrs(v)
            known_ops[attrs['name']].append((getattr(type(obj), k), attrs))

    return known_ops


def find_known_methods(obj: OpResolvable):
    known_methods = set()
    for k in dir(obj):
        v = getattr(obj, k)

        if is_method(v):
            known_methods.add(k)

    return known_methods


def resolve_operator(operators: Dict[str, List[Tuple[Callable, Dict]]], op_info: OpInfo):
    candidates = operators[op_info.op_type]

    #  First filter out downright mismatches
    candidates = [h for h in candidates if h[1]['gen_name'] in [None, op_info.gen_name]]
    candidates = [h for h in candidates if h[1]['gen_group'] in [None, op_info.gen_group]]
    candidates = [h for h in candidates if h[1]['uid'] in [None, op_info.uid]]
    candidates = [h for h in candidates if set(h[1]['tags'] or op_info.tags or []).issuperset(set(op_info.tags or []))]

    #  Get the "most-specific" matches i.e. handlers with the most number of fields specified (not None)
    min_none_cnts = min([list(h[1].values()).count(None) for h in candidates], default=-1)
    candidates = [h for h in candidates if list(h[1].values()).count(None) == min_none_cnts]

    if len(candidates) == 1:
        return candidates[0][0]

    raise ValueError(f"Could not resolve operator handler unambiguously for operator {op_info}.")


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

        index_key = (gen_name, gen_group)
        self.sid_index_map[index_key] += 1
        index = self.sid_index_map[index_key]
        sid = create_sid(gen_name, gen_group, op_type, uid, index)

        return OpInfo(
            sid=sid,
            gen_name=gen_name,
            op_type=op_type,
            index=index,
            uid=uid,
            tags=tuple(tags) if tags is not None else None,
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
