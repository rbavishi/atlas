import ast
import collections
import inspect
import textwrap
import warnings
import weakref
from typing import Callable, Set, Optional, Union, Dict, List, Any, Iterable, Iterator, Type, Tuple

import astunparse

from atlas.hooks import Hook
from atlas.models import GeneratorModel
from atlas.operators import OpInfo, OpInfoConstructor, returns_lambda
from atlas.strategies import RandStrategy, DfsStrategy, PartialReplayStrategy
from atlas.strategy import Strategy
from atlas.tracing import DefaultTracer, GeneratorTrace
from atlas.utils import astutils
from atlas.utils.genutils import register_generator, register_group, get_group_by_name
from atlas.utils.inspection import getclosurevars_recursive
from atlas.warnings import PerformanceWarning
from atlas.wrappers import CallGenerator

_GEN_EXEC_ENV_VAR = "_atlas_gen_exec_env"
_GEN_STRATEGY_VAR = "_atlas_gen_strategy"
_GEN_MODEL_VAR = "_atlas_gen_model"
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

    result = _atlas_gen_strategy.generic_op(*args, **kwargs)

    for h in _atlas_gen_hooks:
        h.after_op(*args, **kwargs, retval=result)

    return result


def cache_wrapper(compiled_func: Callable):
    def wrapper(*args, **kwargs):
        strategy = kwargs.get(_GEN_STRATEGY_VAR)
        is_cached, result = strategy.cached_generator_invocation()
        if is_cached:
            return result

        call_id = strategy.generator_invoked()
        result = compiled_func(*args, **kwargs)
        strategy.generator_returned(call_id, result)
        return result

    return wrapper


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

    if isinstance(strategy, PartialReplayStrategy):
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
    known_methods: Set[str] = strategy.get_known_methods()
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

            n.keywords.append(ast.keyword(arg='model', value=ast.Name(_GEN_MODEL_VAR, ast.Load())))

            n.keywords.append(ast.keyword(arg='op_info', value=ast.Name(f"_op_info_{op_idx}", ast.Load())))
            op_infos[f"_op_info_{op_idx}"] = op_info

            n.keywords.append(ast.keyword(arg='handler', value=ast.Name(f"_handler_{handler_idx}", ast.Load())))
            handler = strategy.get_op_handler(op_info)
            handlers[f"_handler_{handler_idx}"] = handler

            if not with_hooks:
                n.func = astutils.parse(f"{_GEN_STRATEGY_VAR}.generic_op").value
            else:
                n.keywords.append(ast.keyword(arg=_GEN_HOOK_VAR, value=ast.Name(_GEN_HOOK_VAR, ctx=ast.Load())))
                n.keywords.append(ast.keyword(arg=_GEN_STRATEGY_VAR, value=ast.Name(_GEN_STRATEGY_VAR, ctx=ast.Load())))

                n.func.id = _GEN_HOOK_WRAPPER
                ops[_GEN_HOOK_WRAPPER] = hook_wrapper

            if returns_lambda(handler):
                n.func = ast.Call(func=n.func, args=n.args[:], keywords=n.keywords[:])
                n.keywords = []
                n.args = [n.args[0]]

            ast.fix_missing_locations(n)

        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in known_methods:
            #  Similar in spirit to the known_ops case, just much less fancy stuff to do.
            #  Only need to get the right handler which we will achieve by simply making this
            #  a method call instead of a regular call.
            n.func = ast.Attribute(value=ast.Name(_GEN_STRATEGY_VAR, ctx=ast.Load()), attr=n.func.id, ctx=ast.Load())
            ast.fix_missing_locations(n)

        elif isinstance(n, ast.Call):
            #  Try to check if it is a call to a Generator
            #  TODO : Can we be more sophisticated in our static analysis here
            try:
                function = eval(astunparse.unparse(n.func), g)
            except:
                continue

            if isinstance(function, Generator):
                call_id = f"{_GEN_COMPOSITION_ID}_{composition_cnt}"
                composition_cnt += 1
                n.func.id = call_id
                n.keywords.append(ast.keyword(arg=_GEN_EXEC_ENV_VAR,
                                              value=ast.Name(_GEN_EXEC_ENV_VAR, ast.Load())))
                n.keywords.append(ast.keyword(arg=_GEN_STRATEGY_VAR,
                                              value=ast.Name(_GEN_STRATEGY_VAR, ast.Load())))
                n.keywords.append(ast.keyword(arg=_GEN_MODEL_VAR,
                                              value=ast.Name(_GEN_MODEL_VAR, ast.Load())))
                n.keywords.append(ast.keyword(arg=_GEN_HOOK_VAR,
                                              value=ast.Name(_GEN_HOOK_VAR, ast.Load())))
                ast.fix_missing_locations(n)

                #  We delay compilation to handle mutually recursive generators
                delayed_compilations.append((function, call_id))

            elif function is CallGenerator:
                wrapped_func = n.args[0]
                n.func = wrapped_func.func
                n.args = wrapped_func.args[:]
                n.keywords = wrapped_func.keywords[:]
                n.keywords.append(ast.keyword(arg=_GEN_EXEC_ENV_VAR,
                                              value=ast.Name(_GEN_EXEC_ENV_VAR, ast.Load())))
                ast.fix_missing_locations(n)

    #  Add the execution environment argument to the function
    f_ast.args.kwonlyargs.append(ast.arg(arg=_GEN_EXEC_ENV_VAR, annotation=None))
    f_ast.args.kw_defaults.append(ast.NameConstant(value=None))

    #  Add the strategy argument to the function
    f_ast.args.kwonlyargs.append(ast.arg(arg=_GEN_STRATEGY_VAR, annotation=None))
    f_ast.args.kw_defaults.append(ast.NameConstant(value=None))

    #  Add the strategy argument to the function
    f_ast.args.kwonlyargs.append(ast.arg(arg=_GEN_MODEL_VAR, annotation=None))
    f_ast.args.kw_defaults.append(ast.NameConstant(value=None))

    #  Add the hook argument to the function
    f_ast.args.kwonlyargs.append(ast.arg(arg=_GEN_HOOK_VAR, annotation=None))
    f_ast.args.kw_defaults.append(ast.NameConstant(value=None))
    ast.fix_missing_locations(f_ast)

    #  New name so it doesn't clash with original
    func_name = f"{_GEN_COMPILED_TARGET_ID}_{len(cache)}"

    g.update({k: v for k, v in ops.items()})
    g.update({k: v for k, v in handlers.items()})
    g.update({k: v for k, v in op_infos.items()})

    module = ast.Module()
    module.body = [f_ast]

    #  Passing ``g`` to exec allows us to execute all the new functions
    #  we assigned to every operator call in the previous AST walk
    filename = inspect.getabsfile(func)
    exec(compile(module, filename=filename, mode="exec"), g)
    result = g[func.__name__]
    g["__name__"] = filename

    if inspect.ismethod(func):
        result = result.__get__(func.__self__, func.__self__.__class__)

    #  Restore the correct namespace so that tracebacks contain actual function names
    g[gen.name] = gen
    g[func_name] = result

    cache[func] = result

    #  Handle the delayed compilations now that we have populated the cache
    for gen, call_id in delayed_compilations:
        compiled_func = compile_func(gen, gen.func, strategy, with_hooks)
        if gen.caching and isinstance(strategy, DfsStrategy):
            #  Add instructions for using cached result if any
            g[call_id] = cache_wrapper(compiled_func)

        else:
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
                 caching: bool = False,
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
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

        self.caching = caching

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
        Set a model to be used by the generator (strategy).
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
        Invocation during the execution of a top-level generator enables generator composition i.e. the same strategy is
        used to call this generator. If this is the top-level generator, then this is equivalent to the ``.call`` method.

        Args:
            *args: Positional arguments to the original function
            **kwargs: Keyword arguments to the original function
        """

        _atlas_gen_exec_env: GeneratorExecEnvironment = kwargs.pop(_GEN_EXEC_ENV_VAR, None)
        if _atlas_gen_exec_env is not None:
            return _atlas_gen_exec_env.compositional_call(self, args, kwargs)

        #  Try to find the calling generator execution environment
        frame = inspect.currentframe().f_back
        while frame is not None and not ('self' in frame.f_locals and
                                         isinstance(frame.f_locals['self'], GeneratorExecEnvironment)):
            frame = frame.f_back

        if frame is None:
            return self.call(*args, **kwargs)

        #  This is a compositional call so point out the performance problem.
        warnings.warn("Compositional generator invocation discovered at runtime, "
                      "which may incur a performance penalty.\n"
                      "Consider wrapping the call as follows:\n"
                      "from atlas.wrappers import CallGenerator\n"
                      "CallGenerator(...)",
                      PerformanceWarning, stacklevel=2)

        _atlas_gen_exec_env = frame.f_locals['self']
        return _atlas_gen_exec_env.compositional_call(self, args, kwargs)

    def generate(self, *args, **kwargs):
        """
        Create an iterator for the result of all possible executions (all possible combinations of
        operator choices) of the generator function for the given input i.e. ``(*args, **kwargs)``

        Args:
            *args: Positional arguments to the original function
            **kwargs: Keyword arguments to the original function

        Returns:
            An iterator

        """

        if self._default_exec_env is None:
            self._default_exec_env = GeneratorExecEnvironment(
                gen=self,
                strategy=self.strategy,
                model=self.model,
                tracing=False,
                hooks=list(self.hooks),
                replay=None
            )

        yield from self._default_exec_env.generate(*args, **kwargs)

    def call(self, *args, **kwargs):
        """
        Return the value returned by the first execution of the generator for the given input i.e. ``(*args, **kwargs)``.
        This is useful when dealing with randomized generators where iteration is not required.

        Args:
            *args: Positional arguments to the original function
            **kwargs: Keyword arguments to the original function

        Returns:
            Value returned by the first invocation of the generator

        """
        if self._default_exec_env is None:
            self._default_exec_env = GeneratorExecEnvironment(
                gen=self,
                strategy=self.strategy,
                model=self.model,
                tracing=False,
                hooks=list(self.hooks),
                replay=None
            )

        return self._default_exec_env.call(*args, **kwargs)

    def with_env(self,
                 *args,
                 strategy: Union[str, Strategy] = None,
                 model: GeneratorModel = None,
                 tracing: bool = False, hooks: List[Hook] = None,
                 replay: Union[Dict[str, List[Any]], GeneratorTrace] = None,
                 ignore_exceptions: bool = False) -> 'GeneratorExecEnvironment':
        """
        Temporarily modify the config of the generator.

        Args:
            strategy:
            model:
            tracing:
            hooks:
            replay:
            ignore_exceptions:

        Returns:

        """
        if len(args) != 0:
            raise SyntaxError("with_env accepts only keyword arguments")

        return GeneratorExecEnvironment(
            gen=self,
            strategy=make_strategy(strategy or self.strategy),
            model=model or self.model,
            tracing=tracing,
            hooks=list(hooks or self.hooks),
            replay=replay,
            ignore_exceptions=ignore_exceptions
        )

    def __get__(self, instance, owner):
        #  This is required to handle class methods that have been marked as generators.
        #  This helps us create "bound" generators

        #  Create a replica with a different func
        replica = Generator(self.func)
        replica.__dict__ = self.__dict__.copy()
        replica.func = self.func.__get__(instance, owner)
        return replica


class GeneratorExecEnvironment:
    """
    Execution environment for a generator. Provides isolation between simultaneous uses of the generator.
    """

    def __init__(self,
                 gen: Generator,
                 strategy: Strategy,
                 model: Optional[GeneratorModel],
                 tracing: bool,
                 hooks: List[Hook],
                 replay: Optional[Union[Dict[str, List[Any]], GeneratorTrace]],
                 ignore_exceptions: bool = False
                 ):

        self.gen = gen
        self.strategy = strategy
        self.model = model
        self.tracing = tracing
        self.hooks = hooks
        self.replay = replay

        self._compiled_func: Optional[Callable] = None
        self._compilation_cache: Dict[Generator, Callable] = {}
        self.tracer: Optional[DefaultTracer] = None
        self.ignore_exceptions = ignore_exceptions

        self.init()

    def init(self):
        if self.tracing:
            self.tracer = DefaultTracer()
            self.hooks.append(self.tracer)

        if self.replay is not None:
            self.strategy = PartialReplayStrategy(self.replay, self.strategy)

        self._compilation_cache = {}
        self._compiled_func = compile_func(self.gen, self.gen.func, self.strategy, len(self.hooks) > 0)

    def compositional_call(self, gen: Generator, args, kwargs):
        extra_kwargs = {_GEN_EXEC_ENV_VAR: self, _GEN_STRATEGY_VAR: self.strategy,
                        _GEN_MODEL_VAR: self.model, _GEN_HOOK_VAR: self.hooks}

        if gen not in self._compilation_cache:
            self._compilation_cache[gen] = compile_func(gen, gen.func, self.strategy, len(self.hooks) > 0)

        return self.strategy.gen_call(self._compilation_cache[gen], args, kwargs, extra_kwargs, self.gen)

    def generate(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0 and isinstance(self.replay, GeneratorTrace):
            args, kwargs = self.replay.f_inputs

        extra_kwargs = {_GEN_EXEC_ENV_VAR: self, _GEN_STRATEGY_VAR: self.strategy,
                        _GEN_HOOK_VAR: self.hooks, _GEN_MODEL_VAR: self.model}

        iterator = self.strategy.gen_iterate(self._compiled_func, args, kwargs, extra_kwargs,
                                             self.hooks, self.gen, ignore_exceptions=self.ignore_exceptions)
        if self.tracer is None:
            yield from iterator

        else:
            for result in iterator:
                yield result, self.tracer.get_last_trace()

    def call(self, *args, **kwargs):
        return next(self.generate(*args, **kwargs))

    def __call__(self, *args, **kwargs):
        return next(self.generate(*args, **kwargs))


def generator(func=None, strategy='dfs', name=None, group=None, caching=None, metadata=None) -> Generator:
    """Define a generator from a function

    Args:
        func (Callable): The function to define generator

        strategy (Union[str, Strategy]): The strategy to use while executing the generator.
            Can be a string (one of 'dfs' and 'randomized') or an instance of the ``Strategy`` class.
            Default is 'dfs'.

        name (str): Name used to register the generator.
            If unspecified, the generator is not registered.

        group (str): Name of the group to register the generator in.
            If unspecified, the generator is not registered with any group.

        metadata (Dict[Any, Any]): A dictionary containing arbitrary metadata
            to carry around in the generator object.

    Examples:

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

    """
    def wrapper(func):
        return Generator(func, strategy=strategy, name=name, group=group, caching=caching, metadata=metadata)

    if func:
        return wrapper(func)
    else:
        return wrapper

