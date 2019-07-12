from inspect import builtins, ismodule, iscode, ClosureVars


def getclosurevars_recursive(func):
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
            if iscode(const):
                codes.append(const)

    return ClosureVars(nonlocal_vars, global_vars,
                       builtin_vars, unbound_names)
