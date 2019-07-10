import ast
import inspect
from typing import Callable, Set, Optional, Union

import astunparse

from atlas.semantics import Semantics, RandSemantics, DfsSemantics
from atlas.utils import astutils


def get_op_id(n_call: ast.Call) -> Optional[str]:
    for kw in n_call.keywords:
        if kw.arg == 'op_id':
            if not isinstance(kw.value, ast.Str):
                raise Exception("Value passed to 'op_id' must be a string in {}".format(astunparse.unparse(n_call)))

            return kw.value.s

    return None


def make_semantics(semantics: Union[str, Semantics]) -> Optional[Semantics]:
    if semantics is None:
        return None

    if isinstance(semantics, Semantics):
        return semantics

    if semantics == 'randomized':
        return RandSemantics()

    elif semantics == 'dfs':
        return DfsSemantics()

    raise Exception("Unrecognized semantics - {}".format(semantics))


def compile_func(func: Callable, semantics: Semantics):
    f_ast = astutils.parse(inspect.getsource(func))
    f_ast.decorator_list = [d for d in f_ast.decorator_list
                            if (not isinstance(d, ast.Name) or d.id != 'generator') and
                            (not isinstance(d, ast.Attribute) or d.attr != 'generator') and
                            (not (isinstance(d, ast.Call) and isinstance(d.func,
                                                                         ast.Name)) or d.func.id != 'generator')]

    known_ops: Set[str] = semantics.get_known_ops()

    ops = {}
    for n in ast.walk(f_ast):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in known_ops:
            op_kind = n.func.id
            op_id = get_op_id(n)
            new_op_name, op = semantics.process_op(op_kind, op_id)
            n.func.id = new_op_name
            ops[n.func.id] = op

    g = inspect.getclosurevars(func).globals.copy()
    g.update({k: v for k, v in ops.items()})
    exec(astutils.to_source(f_ast), g)
    return g[func.__name__]


class Generator:
    def __init__(self, func: Callable, semantics: Union[str, Semantics] = None, **kwargs):
        self.func = func
        self.semantics: Semantics = make_semantics(semantics)
        self._compiled_func: Callable = None

    def set_semantics(self, semantics: Optional[Union[str, Semantics]] = None):
        self.semantics = make_semantics(semantics)
        self._compiled_func = None

    def __call__(self, *args, **kwargs):
        if self._compiled_func is None:
            self._compiled_func = compile_func(self.func, self.semantics)

        return self._compiled_func(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if self._compiled_func is None:
            self._compiled_func = compile_func(self.func, self.semantics)

        self.semantics.init()
        while not self.semantics.is_finished():
            self.semantics.init_run()
            yield self._compiled_func(*args, **kwargs)
            self.semantics.finish_run()

        self.semantics.finish()


def generator(*args, **kwargs) -> Generator:
    """Define a generator from a function
    """
    allowed_kwargs = {'semantics'}
    error_str = "The @generator decorator should be applied either with no parentheses or " \
                "the following keyword args - semantics."
    assert (len(args) == 1 and len(kwargs) == 0 and callable(args[0])) or \
           (len(args) == 0 and len(kwargs) > 0 and set(kwargs.keys()).issubset(allowed_kwargs)), error_str

    if len(args) == 1:
        result = Generator(args[0])
        result.set_semantics('dfs')
        return result

    else:
        def wrapper(func):
            result = Generator(func, **kwargs)
            result.set_semantics()
            return result

        return wrapper
