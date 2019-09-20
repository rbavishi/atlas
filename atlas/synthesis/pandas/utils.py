import ast
import pandas as pd
from typing import List, Any, Dict


class LambdaWrapper:
    def __init__(self, fn: str = None):
        self.fn = fn

    def __str__(self):
        return self.fn

    def __repr__(self):
        return self.fn

    def __call__(self, *args, **kwargs):
        return eval(self.fn)(*args, **kwargs)


class Program:
    """
    The representation of a Pandas program, used both for training data
    and returning results from the synthesis engine.

    Essentially captures a straight line sequence of function calls along with their arguments
    """

    def __init__(self):
        self.inputs: List[Any] = []
        self.output: Any = None
        self.intermediates: List[Any] = []
        self.arguments: List[Dict] = []
        self.functions: List[str] = []


def create_inversion_template():
    """
    Convenience function to create the gigantic inversion strategy
    """

    api_path = "/Users/rbavishi/Research/Atlas/atlas/synthesis/pandas/api.py"
    cnt = 0

    def get_methods_for_generator(func: ast.FunctionDef):
        nonlocal cnt
        res = ""
        gen_kwargs = {k.arg: k.value for k in func.decorator_list[0].keywords}
        gen_name = gen_kwargs['name'].s

        nodes = []
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and 'uid' in {k.arg for k in node.keywords}:
                uid = [k.value.s for k in node.keywords if k.arg == 'uid'][0]
                nodes.append((uid, node))

        nodes = sorted(nodes, key=lambda x: int(x[0]))
        for uid, node in nodes:
            decorator = f'    @operator(name="{node.func.id}", gen_name="{gen_name}", uid="{uid}")\n'
            func_def = f"    def Inv{cnt}(self, domain, kwargs, **extra_kwargs):\n" \
                       f"        args = self.get_args(state=kwargs)\n\n"
            cnt += 1
            res += decorator + func_def

        return res

    body = ""

    with open(api_path, "r") as f:
        f_ast = ast.parse(f.read())
        for n in ast.walk(f_ast):
            if isinstance(n, ast.FunctionDef) and \
                    len(n.decorator_list) > 0 and n.decorator_list[0].func.id == 'generator':
                body += get_methods_for_generator(n)

    print(body)
