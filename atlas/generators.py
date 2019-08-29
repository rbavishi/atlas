import ast
import collections
import inspect
import sys
import textwrap
import weakref
from typing import Callable, Set, Optional, Union, Dict, List, Any, Iterable, Iterator, Type, Tuple

import astunparse

from atlas.exceptions import ExceptionAsContinue
from atlas.hooks import Hook
from atlas.models import GeneratorModel
from atlas.operators import OpInfo, OpInfoConstructor
from atlas.strategies import Strategy, RandStrategy, DfsStrategy, ReplayStrategy
from atlas.strategies.strategy import IteratorBasedStrategy
from atlas.tracing import DefaultTracer, GeneratorTrace
from atlas.utils import astutils
from atlas.utils.genutils import register_generator, register_group, get_group_by_name
from atlas.utils.inspection import getclosurevars_recursive

_GEN_EXEC_ENV_VAR = "_atlas_gen_exec_env"
_GEN_STRATEGY_VAR = "_atlas_gen_strategy"
_GEN_HOOK_WRAPPER = "_atlas_hook_wrapper"
_GEN_HOOK_VAR = "_atlas_gen_hooks"
_GEN_COMPOSITION_ID = "_atlas_composition_call"
_GEN_COMPILED_TARGET_ID = "_atlas_compiled_function"


def make_strategy(strategy: Union[str, Strategy]) -> Strategy:
    if isinstance(strategy, Strategy):
        return strategy

    if strategy == 'randomized':
        return RandStrategy()

    elif strategy == 'dfs':
        return DfsStrategy()

    raise Exception(f"Unrecognized strategy - {strategy}")


def hook_wrapper(*args, _atlas_gen_strategy=None, _atlas_gen_hooks=None, **kwargs):
    for h in _atlas_gen_hooks:
        h.before_op(*args, **kwargs)

    result = _atlas_gen_strategy.generic_call(*args, **kwargs)

    for h in _atlas_gen_hooks:
        h.after_op(*args, **kwargs, retval=result)

    return result


class CompilationCache:
    WITHOUT_HOOKS: Dict[Type[Strategy], weakref.WeakKeyDictionary] = collections.defaultdict(weakref.WeakKeyDictionary)
    WITH_HOOKS: Dict[Type[Strategy], weakref.WeakKeyDictionary] = collections.defaultdict(weakref.WeakKeyDictionary)


def compile_func(gen: 'Generator', func: Callable, strategy: Strategy, with_hooks: bool = False) -> Callable:
    """
    The compilation basically assigns functionality to each of the operator calls as
    governed by the semantics (strategy). Memoization is done with the keys as the `func`,
    the class of the `strategy` and the `with_hooks` argument.

    Args:
        gen (Generator): The generator object containing the function to compile
        func (Callable): The function to compile
        strategy (Strategy): The strategy governing the behavior of the operators
        with_hooks (bool): Whether support for hooks is required

    Returns:
        The compiled function

    """

    if isinstance(strategy, ReplayStrategy):
        strategy = strategy.backup_strategy

    if with_hooks:
        cache = CompilationCache.WITH_HOOKS[strategy.__class__]
    else:
        cache = CompilationCache.WITHOUT_HOOKS[strategy.__class__]

    if func in cache:
        return cache[func]

    cache[func] = None

    source_code, start_lineno = inspect.getsourcelines(func)
    source_code = ''.join(source_code)
    f_ast = astutils.parse(textwrap.dedent(source_code))

    # This matches up line numbers with original file and is thus super useful for debugging
    ast.increment_lineno(f_ast, start_lineno - 1)

    #  Remove the ``@generator`` decorator to avoid recursive compilation
    f_ast.decorator_list = [d for d in f_ast.decorator_list
                            if (not isinstance(d, ast.Name) or d.id != 'generator') and
                            (not isinstance(d, ast.Attribute) or d.attr != 'generator') and
                            (not (isinstance(d, ast.Call) and isinstance(d.func,
                                                                         ast.Name)) or d.func.id != 'generator')]

    #  Get all the external dependencies of this function.
    #  We rely on a modified closure function adopted from the ``inspect`` library.
    closure_vars = getclosurevars_recursive(func, f_ast)
    g = {**closure_vars.nonlocals.copy(), **closure_vars.globals.copy()}
    known_ops: Set[str] = strategy.get_known_ops()
    op_info_constructor = OpInfoConstructor()
    delayed_compilations: List[Tuple[Generator, str]] = []

    ops = {}
    handlers = {}
    op_infos = {}
    op_idx: int = 0
    composition_cnt: int = 0
    for n in astutils.preorder_traversal(f_ast):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in known_ops:
            #  Rename the function call, and assign a new function to be called during execution.
            #  This new function is determined by the semantics (strategy) being used for compilation.
            #  Also determine if there any eligible hooks for this operator call.
            op_idx += 1
            handler_idx = len(handlers)
            op_info: OpInfo = op_info_constructor.get(n, gen.name, gen.group)

            n.keywords.append(ast.keyword(arg='op_info', value=ast.Name(f"_op_info_{op_idx}", ctx=ast.Load())))
            op_infos[f"_op_info_{op_idx}"] = op_info

            n.keywords.append(ast.keyword(arg='handler', value=ast.Name(f"_handler_{handler_idx}", ctx=ast.Load())))
            handler = strategy.get_op_handler(op_info)
            handlers[f"_handler_{handler_idx}"] = handler

            if not with_hooks:
                n.func = astutils.parse(f"{_GEN_STRATEGY_VAR}.generic_call").value
            else:
                n.keywords.append(ast.keyword(arg=_GEN_HOOK_VAR, value=ast.Name(_GEN_HOOK_VAR, ctx=ast.Load())))
                n.keywords.append(ast.keyword(arg=_GEN_STRATEGY_VAR, value=ast.Name(_GEN_STRATEGY_VAR, ctx=ast.Load())))

                n.func.id = _GEN_HOOK_WRAPPER
                ops[_GEN_HOOK_WRAPPER] = hook_wrapper

            ast.fix_missing_locations(n)

        elif isinstance(n, ast.Call):
            #  Try to check if it is a call to a Generator
            #  TODO : Can we be more sophisticated in our static analysis here
            try:
                function = eval(astunparse.unparse(n.func), g)
                if isinstance(function, Generator):
                    call_id = f"{_GEN_COMPOSITION_ID}_{composition_cnt}"
                    composition_cnt += 1
                    n.func.id = call_id
                    n.keywords.append(ast.keyword(arg=_GEN_EXEC_ENV_VAR,
                                                  value=ast.Name(_GEN_EXEC_ENV_VAR, ctx=ast.Load())))
                    n.keywords.append(ast.keyword(arg=_GEN_STRATEGY_VAR,
                                                  value=ast.Name(_GEN_STRATEGY_VAR, ctx=ast.Load())))
                    n.keywords.append(ast.keyword(arg=_GEN_HOOK_VAR,
                                                  value=ast.Name(_GEN_HOOK_VAR, ctx=ast.Load())))
                    ast.fix_missing_locations(n)

                    #  We delay compilation to handle mutually recursive generators
                    delayed_compilations.append((function, call_id))

            except:
                pass

    #  Add the execution environment argument to the function
    f_ast.args.args.append(ast.arg(arg=_GEN_EXEC_ENV_VAR, annotation=None))
    f_ast.args.defaults.append(ast.NameConstant(value=None))

    #  Add the strategy argument to the function
    f_ast.args.args.append(ast.arg(arg=_GEN_STRATEGY_VAR, annotation=None))
    f_ast.args.defaults.append(ast.NameConstant(value=None))

    #  Add the strategy argument to the function
    f_ast.args.args.append(ast.arg(arg=_GEN_HOOK_VAR, annotation=None))
    f_ast.args.defaults.append(ast.NameConstant(value=None))
    ast.fix_missing_locations(f_ast)

    #  Change name so it doesn't clash with original
    func_name = f"{_GEN_COMPILED_TARGET_ID}_{len(cache)}"
    f_ast.name = func_name

    g.update({k: v for k, v in ops.items()})
    g.update({k: v for k, v in handlers.items()})
    g.update({k: v for k, v in op_infos.items()})

    module = ast.Module()
    module.body = [f_ast]

    #  Passing ``g`` to exec allows us to execute all the new functions
    #  we assigned to every operator call in the previous AST walk
    exec(compile(module, filename=inspect.getabsfile(func), mode="exec"), g)
    result = g[func_name]
    cache[func] = result

    #  Handle the delayed compilations now that we have populated the cache
    for gen, call_id in delayed_compilations:
        compiled_func = compile_func(gen, gen.func, strategy, with_hooks)
        g[call_id] = compiled_func

    return result


class Generator:
    """
    The result of applying the ``@generator`` decorator to functions is an instance
    of this class, which can then be used to run generators as well as modify their behaviors.
    """

    def __init__(self,
                 func: Callable,
                 strategy: Union[str, Strategy] = 'dfs',
                 model: GeneratorModel = None,
                 name: str = None,
                 group: str = None,
                 metadata: Dict[Any, Any] = None,
                 **kwargs):

        if not inspect.isfunction(func):
            raise TypeError("func is not a Function object")

        self.func = func
        self.strategy: Strategy = make_strategy(strategy)
        self.model = model

        self.name = name or func.__name__
        if name is not None:
            register_generator(self, name)

        self.group = group
        if group is not None:
            register_group(self, group)

        self.hooks: List[Hook] = []
        self.metadata = metadata

        self._default_exec_env: Optional[GeneratorExecEnvironment] = None

    def set_default_strategy(self, strategy: Union[str, Strategy], as_group: bool = True):
        """
        Set a new strategy for the generator. This is useful for exploring different behaviors of the generator
        without redefining the function.

        Args:
            strategy (Union[str, Strategy]): The new strategy to set.
            as_group (bool): Whether to set this strategy for all the generators in the group (if any).
                ``True`` by default.

        """
        self.strategy = make_strategy(strategy)

        if as_group and self.group is not None:
            for g in get_group_by_name(self.group):
                if g is not self:
                    g.set_strategy(self.strategy, as_group=False)

        self._default_exec_env = None

    def set_default_model(self, model: GeneratorModel):
        """
        Set a model to be used by the generator (strategy). Note that a model can work only with a strategy
        that is an instance of the IteratorBasedStrategy abstract class. DfsStrategy is an example of such a strategy
        Args:
            model (GeneratorModel): The model to use

        """
        self.model = model
        self._default_exec_env = None

    def register_default_hooks(self, *hooks: Hook, as_group: bool = True):
        """
        Register hooks for the generator.

        Args:
            *hooks (Hook): The list of hooks to register
            as_group (bool): Whether to set this strategy for all the generators in the group (if any).
                ``True`` by default.

        """
        self.hooks.extend(hooks)

        if as_group and self.group is not None:
            for g in get_group_by_name(self.group):
                if g is not self:
                    g.register_hooks(*hooks, as_group=False)

    def deregister_default_hook(self, hook: Hook, as_group: bool = True, ignore_errors: bool = False):
        """
        De-register hook for the generator.

        Args:
            hook (Hook): The list of hooks to register
            as_group (bool): Whether to de-register this hook for all the generators in the group (if any).
                ``True`` by default.
            ignore_errors (bool): Do not raise exception if the hook was not registered before

        """
        if all(i is not hook for i in self.hooks):
            if not ignore_errors:
                raise ValueError("Hook was not registered.")

        else:
            self.hooks.remove(hook)

        if as_group and self.group is not None:
            for g in get_group_by_name(self.group):
                if g is not self:
                    g.deregister_hook(hook, as_group=False, ignore_errors=True)

    def __call__(self, *args, **kwargs):
        """Functions with an ``@generator`` annotation can be called as any regular function as a result of this method.
        In case of deterministic strategies such as DFS, this will return first possible value. For model-backed
        strategies, the generator will return the value corresponding to an execution path where all the operators
        make the choice with the highest probability as directed by their respective models.

        Args:
            *args: Positional arguments to the original function
            **kwargs: Keyword arguments to the original function
        """

        _atlas_gen_exec_env: GeneratorExecEnvironment = kwargs.get(_GEN_EXEC_ENV_VAR, None)
        if _atlas_gen_exec_env is None:
            #  TODO : Add a one-time performance warning
            frame = inspect.currentframe().f_back
            while not ('self' in frame.f_locals and isinstance(frame.f_locals['self'], GeneratorExecEnvironment)):
                frame = frame.f_back

            _atlas_gen_exec_env = frame.f_locals['self']

        return _atlas_gen_exec_env.compositional_call(self, self.func, args, kwargs)

    def generate(self, *args, **kwargs):
        """
        Create an iterator for the result of all possible executions (all possible combinations of
        operator choices) of the generator function for the given input i.e. ``(*args, **kwargs)``

        Args:
            *args: Positional arguments to the original function
            **kwargs: Keyword arguments to the original function

        Returns:
            An instance of GeneratorExecEnvironment (which subclasses Iterable)

        """

        return GeneratorExecEnvironment(
            args=args,
            kwargs=kwargs,
            func=self.func,
            strategy=self.strategy,
            model=self.model,
            hooks=list(self.hooks),
            parent_gen=self
        )

    def call(self, *args, **kwargs):
        """
        Return the value returned by the frst execution of the generator for the given input i.e. ``(*args, **kwargs)``.
        This is useful when dealing with randomized generators where iteration is not required.

        Args:
            *args: Positional arguments to the original function
            **kwargs: Keyword arguments to the original function

        Returns:
            Value returned by the first invocation of the generator

        """

        if self._default_exec_env is None:
            self._default_exec_env = GeneratorExecEnvironment(
                args=args,
                kwargs=kwargs,
                func=self.func,
                strategy=self.strategy,
                model=self.model,
                hooks=list(self.hooks),
                parent_gen=self
            )

        return self._default_exec_env.single_call(args, kwargs)


class GeneratorExecEnvironment(Iterable):
    """
    The result of calling ``generate(...)`` on a Generator object.
    """

    def __init__(self,
                 args: Optional,
                 kwargs: Optional[Dict],
                 func: Callable,
                 strategy: Strategy,
                 model: Optional[GeneratorModel],
                 hooks: List[Hook],
                 parent_gen: Generator):

        self.args = args
        self.kwargs = kwargs
        self.func = func
        self.strategy = strategy
        self.model = model
        self.hooks = hooks
        self.parent_gen: Generator = parent_gen

        self._compiled_func: Optional[Callable] = None
        self._compilation_cache: Dict[Generator, Callable] = {}
        self.tracer: Optional[DefaultTracer] = None

        self.extra_kwargs = {}
        self.reset_compilation()

    def reset_compilation(self):
        self._compiled_func = None
        self._compilation_cache = {}

        if self.model is not None and isinstance(self.strategy, IteratorBasedStrategy):
            self.strategy.set_model(self.model)

        if self._compiled_func is None:
            self._compiled_func = compile_func(self.parent_gen, self.func, self.strategy, len(self.hooks) > 0)

    def compositional_call(self, parent_gen: Generator, func: Callable, args, kwargs):
        extra_kwargs = {_GEN_EXEC_ENV_VAR: self, _GEN_STRATEGY_VAR: self.strategy, _GEN_HOOK_VAR: self.hooks}

        if parent_gen not in self._compilation_cache:
            self._compilation_cache[parent_gen] = compile_func(parent_gen, func, self.strategy, len(self.hooks) > 0)

        return self._compilation_cache[parent_gen](*args, **kwargs, **extra_kwargs)

    def single_call(self, args, kwargs):
        extra_kwargs = {_GEN_EXEC_ENV_VAR: self, _GEN_STRATEGY_VAR: self.strategy, _GEN_HOOK_VAR: self.hooks}

        self.strategy.init()
        while not self.strategy.is_finished():
            self.strategy.init_run()
            try:
                result = self._compiled_func(*args, **kwargs, **extra_kwargs)
                self.strategy.finish_run()
                self.strategy.finish()
                return result

            except ExceptionAsContinue:
                pass

            finally:
                self.strategy.finish_run()

    def __iter__(self) -> Iterator:
        extra_kwargs = {_GEN_EXEC_ENV_VAR: self, _GEN_STRATEGY_VAR: self.strategy, _GEN_HOOK_VAR: self.hooks}

        for h in self.hooks:
            h.init(self.args, self.kwargs)

        self.strategy.init()
        while not self.strategy.is_finished():
            for h in self.hooks:
                h.init_run(self.args, self.kwargs)

            self.strategy.init_run()
            try:
                result = self._compiled_func(*self.args, **self.kwargs, **extra_kwargs)
                if self.tracer is None:
                    yield result
                else:
                    yield result, self.tracer.get_last_trace()

            except ExceptionAsContinue:
                pass

            self.strategy.finish_run()

            for h in self.hooks:
                h.finish_run()

        self.strategy.finish()

        for h in self.hooks:
            h.finish()

    def with_tracing(self) -> 'GeneratorExecEnvironment':
        """
        Enable tracing in this environment. During iteration, the trace of the choices made by the operators
        along with some other meta-data will be returned alongside the result of the generator.

        Returns:
            The same GeneratorExecEnvironment object (self) to enable chaining of ``with_*`` calls

        """
        self.tracer = DefaultTracer()
        self.hooks.append(self.tracer)
        self.reset_compilation()
        return self

    def with_hooks(self, *hooks: Hook) -> 'GeneratorExecEnvironment':
        """
         Register hooks in this environment. This is useful if you want to register hooks temporarily for one
         particular ``.generate(...)`` call of a Generator object without resetting the default hooks of the Generator.

         Args:
             *hooks (Hook): The hooks to add to the environment

         Returns:
             The same GeneratorExecEnvironment object (self) to enable chaining of ``with_*`` calls

         """
        self.hooks.extend(hooks)
        self.reset_compilation()
        return self

    def with_strategy(self, strategy: Union[str, Strategy]) -> 'GeneratorExecEnvironment':
        """
        Set the strategy to be used in this environment. This is useful if you want to use a different
        strategy temporarily for one particular ``.generate(...)`` call of a Generator object
        without resetting the default strategy for the Generator.

        Args:
            strategy (Union[str, Strategy]): The strategy to set for this particular environment.

        Returns:
             The same GeneratorExecEnvironment object (self) to enable chaining of ``with_*`` calls

        """
        self.strategy = make_strategy(strategy)
        self.reset_compilation()
        return self

    def with_model(self, model: GeneratorModel) -> 'GeneratorExecEnvironment':
        """
        Set the model to be used in this environment. This is useful if you want to use a different
        model temporarily for one particular ``.generate(...)`` call of a Generator object
        without resetting the default model for the Generator.

        Args:
            model (GeneratorModel): The model to use in this particular environment.

        Returns:
             The same GeneratorExecEnvironment object (self) to enable chaining of ``with_*`` calls

        """
        self.model = model
        return self

    def replay(self, trace: Union[Dict[str, List[Any]], GeneratorTrace]):
        """
        Replay the choices made by the operators in a trace. The trace can either be a GeneratorTrace object,
        or a mapping/dict from operator labels to a list of values. Operators with the same label will consume
        values from the corresponding list in execution order. If labels are unique (recommended practice),
        the list should be a singleton.

        Args:
            trace (Union[Dict[str, List[Any]], GeneratorTrace]): The trace to be replayed

        Returns:
             The same GeneratorExecEnvironment object (self) to enable chaining of ``with_*`` calls
        """
        if not self.args and not self.kwargs:
            self.args, self.kwargs = trace.f_inputs
        self.strategy = ReplayStrategy(trace, self.strategy)
        return self


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
    error_str = f"The @generator decorator should be applied either with no parentheses or " \
                f"at least one of the following keyword args - {', '.join(allowed_kwargs)}."
    assert (len(args) == 1 and len(kwargs) == 0 and callable(args[0])) or \
           (len(args) == 0 and len(kwargs) > 0 and set(kwargs.keys()).issubset(allowed_kwargs)), error_str

    if len(args) == 1:
        return Generator(args[0])

    else:
        def wrapper(func):
            return Generator(func, **kwargs)

        return wrapper
