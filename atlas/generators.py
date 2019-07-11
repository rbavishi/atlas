import ast
import inspect
from typing import Callable, Set, Optional, Union, Dict, List, Any

import astunparse

from atlas.semantics import Semantics, RandSemantics, DfsSemantics
from atlas.semantics.base import PyGeneratorBasedSemantics
from atlas.utils import astutils
from atlas.utils.genutils import register_generator, register_group, get_group_by_name


def get_op_id(n_call: ast.Call) -> Optional[str]:
    for kw in n_call.keywords:
        if kw.arg == 'op_id':
            if not isinstance(kw.value, ast.Str):
                raise Exception("Value passed to 'op_id' must be a string in {}".format(astunparse.unparse(n_call)))

            return kw.value.s

    return None


def make_semantics(semantics: Union[str, Semantics]) -> Semantics:
    if isinstance(semantics, Semantics):
        return semantics

    if semantics == 'randomized':
        return RandSemantics()

    elif semantics == 'dfs':
        return DfsSemantics()

    raise Exception("Unrecognized semantics - {}".format(semantics))


def convert_func_to_python_generator(f_ast: ast.FunctionDef, semantics: Semantics) -> ast.FunctionDef:
    raise NotImplementedError


def compile_func(func: Callable, semantics: Semantics) -> Callable:
    f_ast = astutils.parse(inspect.getsource(func))
    f_ast.decorator_list = [d for d in f_ast.decorator_list
                            if (not isinstance(d, ast.Name) or d.id != 'generator') and
                            (not isinstance(d, ast.Attribute) or d.attr != 'generator') and
                            (not (isinstance(d, ast.Call) and isinstance(d.func,
                                                                         ast.Name)) or d.func.id != 'generator')]

    g = inspect.getclosurevars(func).globals.copy()

    if isinstance(semantics, PyGeneratorBasedSemantics):
        f_ast = convert_func_to_python_generator(f_ast, semantics)

    else:
        known_ops: Set[str] = semantics.get_known_ops()

        ops = {}
        for n in ast.walk(f_ast):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in known_ops:
                op_kind = n.func.id
                op_id = get_op_id(n)
                new_op_name, op = semantics.process_op(op_kind, op_id)
                n.func.id = new_op_name
                ops[n.func.id] = op

        g.update({k: v for k, v in ops.items()})

    exec(astutils.to_source(f_ast), g)
    return g[func.__name__]


class Generator:
    def __init__(self,
                 func: Callable,
                 semantics: Union[str, Semantics] = 'dfs',
                 name: str = None,
                 group: str = None,
                 metadata: Dict[Any, Any] = None,
                 **kwargs):

        self.func = func
        self.semantics: Semantics = make_semantics(semantics)
        self._compiled_func: Callable = None

        self.name = name
        if name is not None:
            register_generator(self, name)

        self.group = group
        if group is not None:
            try:
                gen_group = get_group_by_name(group)
                self.semantics = gen_group[0].semantics

            except KeyError:
                pass

            register_group(self, group)

        self.metadata = metadata

    def set_semantics(self, semantics: Union[str, Semantics], as_group: bool = True):
        self.semantics = make_semantics(semantics)
        self._compiled_func = None

        if as_group and self.group is not None:
            for g in get_group_by_name(self.group):
                g.set_semantics(self.semantics, as_group=False)

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

    Can be used with no arguments or specific keyword arguments to define a generator as follows:

    .. code-block:: python

        from atlas import generator

        @generator
        def g(length):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

    The generator above can be used to enumerate all binary strings of length ``2`` as follows

    .. code-block:: python

        for s in g.generate(2):
            print(s)

    with the output as

    .. code-block:: python

        00
        01
        10
        11

    The function also accepts specific keyword arguments:

    * **semantics:** The semantics to use while executing the generator.
    * **name:** Name used to register the generator. If unspecified, the generator is not registered.
    * **group:** Name of the group to register the generator in. If unspecified,
      the generator is not registered with any group.
    * **metadata:** A dictionary containing arbitrary metadata to
      carry around in the generator object.

    """
    allowed_kwargs = {'semantics', 'name', 'group', 'metadata'}
    error_str = "The @generator decorator should be applied either with no parentheses or " \
                "at least one of the following keyword args - {}.".format(', '.join(allowed_kwargs))
    assert (len(args) == 1 and len(kwargs) == 0 and callable(args[0])) or \
           (len(args) == 0 and len(kwargs) > 0 and set(kwargs.keys()).issubset(allowed_kwargs)), error_str

    if len(args) == 1:
        return Generator(args[0])

    else:
        def wrapper(func):
            return Generator(func, **kwargs)

        return wrapper
