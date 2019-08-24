from inspect import builtins, ismodule, iscode, ClosureVars


def getclosurevars_recursive(func):
    """
    The default getclosurevars doesn't go over nested function defs and list comprehensions.
    We write a recursive version of the same.
    The logic is borrowed from this post - https://bugs.python.org/issue34947
    Args:
        func (Callable): The function to inspect

    Returns:
        An instance of ClosureVars

    """
    f_code = func.__code__
    # Nonlocal references are named in co_freevars and resolved
    # by looking them up in __closure__ by positional index
    if func.__closure__ is None:
        nonlocal_vars = {}
    else:
        nonlocal_vars = {
            var: cell.cell_contents
            for var, cell in zip(f_code.co_freevars, func.__closure__)
        }

    # Global and builtin references are named in co_names and resolved
    # by looking them up in __globals__ or __builtins__
    global_ns = func.__globals__
    builtin_ns = global_ns.get("__builtins__", builtins.__dict__)
    if ismodule(builtin_ns):
        builtin_ns = builtin_ns.__dict__
    global_vars = {}
    builtin_vars = {}
    unbound_names = set()
    codes = [f_code]
    while codes:
        #  The logic is recursive but is implemented iteratively
        code = codes.pop()
        for name in code.co_names:
            if name in ("None", "True", "False"):
                # Because these used to be builtins instead of keywords, they
                # may still show up as name references. We ignore them.
                continue
            try:
                global_vars[name] = global_ns[name]
            except KeyError:
                try:
                    builtin_vars[name] = builtin_ns[name]
                except KeyError:
                    unbound_names.add(name)

        for const in code.co_consts:
            #  Add the code to inspect recursively
            if iscode(const):
                codes.append(const)

    #  Also check for annotations. This is not natively handled by the current getclosurevars implementation of inspect
    for k, v in func.__annotations__.items():
        try:
            global_vars[v.__name__] = v
        except AttributeError:
            pass

    return ClosureVars(nonlocal_vars, global_vars,
                       builtin_vars, unbound_names)
