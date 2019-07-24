import ast
import inspect
import textwrap
from typing import Callable, Set, Optional, Union, Dict, List, Any, Tuple

import astunparse

from atlas.hooks import PreHook, PostHook, Hook
from atlas.strategies import Strategy, RandStrategy, DfsStrategy
from atlas.utils import astutils
from atlas.utils.genutils import register_generator, register_group, get_group_by_name
from atlas.utils.inspection import getclosurevars_recursive
from atlas.exceptions import ExceptionAsContinue


def get_sid(n_call: ast.Call) -> Optional[str]:
    for kw in n_call.keywords:
        if kw.arg == 'sid':
            if not isinstance(kw.value, ast.Str):
                raise Exception("Value passed to 'sid' must be a string in {}".format(astunparse.unparse(n_call)))

            return kw.value.s

    return None


def make_strategy(strategy: Union[str, Strategy]) -> Strategy:
    if isinstance(strategy, Strategy):
        return strategy

    if strategy == 'randomized':
        return RandStrategy()

    elif strategy == 'dfs':
        return DfsStrategy()

    raise Exception("Unrecognized strategy - {}".format(strategy))


def compile_op(op: Callable, pre_hooks: List[Callable], post_hooks: List[Callable]) -> Callable:
    """
    Returns the op as is if there are no hooks to register. Otherwise, creates a closure that executes the hooks
    in the appropriate order before returning the result of the operator (as defined by the strategy
    passed to compile_func)
    Args:
        op (Callable): The operator call returned by the strategy
        pre_hooks (List[Callable]): The pre-hooks to register
        post_hooks (List[Callable]): The post-hooks to register

    Returns:
        A callable that executes the hooks (if any) along with the operator and returns result of the operator call
    """
    if len(pre_hooks) == 0 and len(post_hooks) == 0:
        return op

    def create_op(*args, **kwargs):
        for f in pre_hooks:
            f(*args, **kwargs)

        result = op(*args, **kwargs)

        for f in post_hooks:
            f(*args, **kwargs, retval=result)

        return result

    return create_op


def compile_func(func: Callable, strategy: Strategy,
                 pre_hooks: List[PreHook] = None, post_hooks: List[PostHook] = None) -> Callable:
    """
    The compilation basically assigns functionality to each of the operator calls as
    governed by the semantics (strategy).

    Args:
        func (Callable): The function to compile
        strategy (Strategy): The strategy governing the behavior of the operators
        pre_hooks (List[PreHook]): A list of hooks that need to be executed before an operator call
        post_hooks (List[PostHook]): A list of hooks that need to be executed after an operator call.
            These hooks also receive the return value of the operator call as a keyword argument 'retval'.

    Returns:
        The compiled function

    """

    if pre_hooks is None:
        pre_hooks = []
    if post_hooks is None:
        post_hooks = []

    source_code, start_lineno = inspect.getsourcelines(func)
    source_code = ''.join(source_code)
    f_ast = astutils.parse(textwrap.dedent(source_code))

    # This matches up line numbers with original file and is thus super useful for debugging
    ast.increment_lineno(f_ast, start_lineno - 1)

    #  Remove the ``@generator`` decorator to avsid recursive compilation
    f_ast.decorator_list = [d for d in f_ast.decorator_list
                            if (not isinstance(d, ast.Name) or d.id != 'generator') and
                            (not isinstance(d, ast.Attribute) or d.attr != 'generator') and
                            (not (isinstance(d, ast.Call) and isinstance(d.func,
                                                                         ast.Name)) or d.func.id != 'generator')]

    #  Get all the external dependencies of this function.
    #  We rely on a modified closure function adopted from the ``inspect`` library.
    g = getclosurevars_recursive(func).globals.copy()
    known_ops: Set[str] = strategy.get_known_ops()

    ops = {}
    for n in astutils.preorder_traversal(f_ast):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in known_ops:
            #  Rename the function call, and assign a new function to be called during execution.
            #  This new function is determined by the semantics (strategy) being used for compilation.
            #  Also determine if there any eligible hooks for this operator call.
            op_name = n.func.id
            sid = get_sid(n)
            new_op_name, sid, op = strategy.process_op(op_name, sid)
            op_pre_hooks = [x for x in [hook.create_hook(op_name, sid) for hook in pre_hooks] if x is not None]
            op_post_hooks = [x for x in [hook.create_hook(op_name, sid) for hook in post_hooks] if x is not None]

            n.func.id = new_op_name
            ops[n.func.id] = compile_op(op, op_pre_hooks, op_post_hooks)

    g.update({k: v for k, v in ops.items()})

    module = ast.Module()
    module.body = [f_ast]

    #  Passing ``g`` to exec allows us to execute all the new functions
    #  we assigned to every operator call in the previous AST walk
    exec(compile(module, filename=inspect.getabsfile(func), mode="exec"), g)
    return g[func.__name__]


class Generator:
    """
    The result of applying the ``@generator`` decorator to functions is an instance
    of this class, which can then be used to run generators as well as modify their behaviors.
    """
    def __init__(self,
                 func: Callable,
                 strategy: Union[str, Strategy] = 'dfs',
                 name: str = None,
                 group: str = None,
                 metadata: Dict[Any, Any] = None,
                 **kwargs):

        self.func = func
        self.strategy: Strategy = make_strategy(strategy)
        self._compiled_func: Callable = None

        self.name = name
        if name is not None:
            register_generator(self, name)

        self.group = group
        if group is not None:
            try:
                gen_group = get_group_by_name(group)
                #  Generators in the same group share their strategies by default
                self.strategy = gen_group[0].strategy

            except KeyError:
                pass

            register_group(self, group)

        self.pre_hooks: List[PreHook] = []
        self.post_hooks: List[PostHook] = []
        self.metadata = metadata

    def set_strategy(self, strategy: Union[str, Strategy], as_group: bool = True):
        """
        Set a new strategy for the generator. This is useful for exploring different behaviors of the generator
        without redefining the function.

        Args:
            strategy (Union[str, Strategy]): The new strategy to set.
            as_group (bool): Whether to set this strategy for all the generators in the group (if any).
                ``True`` by default.

        """
        self.strategy = make_strategy(strategy)
        self._compiled_func = None

        if as_group and self.group is not None:
            for g in get_group_by_name(self.group):
                g.set_strategy(self.strategy, as_group=False)

    def register_hooks(self, *hooks: Hook, as_group: bool = True):
        """
        Register hooks for the generator. Hooks are functions that execute before (pre-hooks) or after (post-hooks)
        every operator call. Hooks can contain operator-specific behavior (just like Strategies) enabling a myriad of
        utilities such as tracing and debugging.

        Args:
            *hooks (Hook): The list of hooks to register
            as_group (bool): Whether to set this strategy for all the generators in the group (if any).
                ``True`` by default.

        """
        self.pre_hooks.extend([h for h in hooks if isinstance(h, PreHook)])
        self.post_hooks.extend([h for h in hooks if isinstance(h, PostHook)])
        self._compiled_func = None

        if as_group and self.group is not None:
            for g in get_group_by_name(self.group):
                g.register_hooks(*hooks, as_group=False)

    def __call__(self, *args, **kwargs):
        """Functions with an ``@generator`` annotation can be called as any regular function as a result of this method.
        In case of deterministic strategies such as DFS, this will return first possible value. For model-backed
        strategies, the generator will return the value corresponding to an execution path where all the operators
        make the choice with the highest probability as directed by their respective models.

        Args:
            *args: Positional arguments to the original function
            **kwargs: Keyword arguments to the original function
        """
        if self._compiled_func is None:
            self._compiled_func = compile_func(self.func, self.strategy, self.pre_hooks, self.post_hooks)

        return self._compiled_func(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        This method returns an iterator for the result of all possible executions (all possible combinations of
        operator choices) of the generator function for the given input i.e. ``(*args, **kwargs)``

        Args:
            *args: Positional arguments to the original function
            **kwargs: Keyword arguments to the original function

        Returns:
            An iterator for all the possible values that can be returned by the generator function.

        """
        if self._compiled_func is None:
            self._compiled_func = compile_func(self.func, self.strategy, self.pre_hooks, self.post_hooks)

        self.strategy.init()
        while not self.strategy.is_finished():
            self.strategy.init_run()
            try:
                yield self._compiled_func(*args, **kwargs)

            except ExceptionAsContinue:
                pass

            self.strategy.finish_run()

        self.strategy.finish()


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

    Keyword Args:
        strategy (Union[str, Strategy]): The strategy to use while executing the generator.
            Can be a string (one of 'dfs' and 'randomized') or an instance of the ``Strategy`` class.
            Default is 'dfs'.

        name (str): Name used to register the generator.
            If unspecified, the generator is not registered.

        group (str): Name of the group to register the generator in.
            If unspecified, the generator is not registered with any group.

        metadata (Dict[Any, Any]): A dictionary containing arbitrary metadata
            to carry around in the generator object.
    """
    allowed_kwargs = {'strategy', 'name', 'group', 'metadata'}
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
