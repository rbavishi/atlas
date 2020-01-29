import itertools
import unittest
from typing import Any

from atlas import generator
from atlas.exceptions import ExceptionAsContinue
from atlas.models import GeneratorModel
from atlas.operators import operator, method, OpInfo
from atlas.strategies import DfsStrategy
from atlas.utils.stubs import stub
from atlas.warnings import PerformanceWarning
from atlas.wrappers import CallGenerator


@stub
def Select(*args, **kwargs):
    pass


@stub
def SelectReversed(*args, **kwargs):
    pass


@stub
def SomeRandomMethod(*args, **kwargs):
    pass


class TestBasicGeneratorFunctionality(unittest.TestCase):
    def test_gen_single_1(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        self.assertEqual(list(binary.generate(2)), ["00", "01", "10", "11"])

    def test_gen_single_2(self):
        @generator
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        self.assertEqual(list(binary.with_env(strategy='dfs').generate(2)), ["00", "01", "10", "11"])

    def test_gen_custom_strategy_1(self):
        class ReversedDFS(DfsStrategy):
            @operator(name='Select', uid="reversed")
            def Select_reversed(self, domain, *args, **kwargs):
                yield from reversed(domain)

        @generator(strategy=ReversedDFS())
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"], uid='reversed')

            return s

        self.assertEqual(list(binary.generate(2)), list(reversed(["00", "01", "10", "11"])))

    def test_gen_custom_strategy_2(self):
        class ReversedDFS(DfsStrategy):
            @operator(name='Select', uid="reversed")
            def Select_reversed(self, domain, *args, **kwargs):
                yield from reversed(domain)

        @generator
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"], uid='reversed')

            return s

        self.assertEqual(list(binary.with_env(strategy=ReversedDFS()).generate(2)),
                         list(reversed(["00", "01", "10", "11"])))

    def test_gen_custom_strategy_3(self):
        class ReversedDFS(DfsStrategy):
            @operator
            def SelectReversed(self, domain, *args, **kwargs):
                yield from reversed(domain)

        @generator(strategy=ReversedDFS())
        def binary(length: int):
            s = ""
            for i in range(length):
                s += SelectReversed(["0", "1"])

            return s

        self.assertEqual(list(binary.generate(2)), list(reversed(["00", "01", "10", "11"])))

    def test_gen_custom_strategy_4(self):
        shared_var = []

        class ReversedDFS(DfsStrategy):
            @operator
            def SelectReversed(self, domain, *args, **kwargs):
                yield from reversed(domain)

            @method
            def SomeRandomMethod(self, key: int):
                shared_var.append(key)

        @generator(strategy=ReversedDFS())
        def binary(length: int):
            s = ""
            for i in range(length):
                s += SelectReversed(["0", "1"])

            SomeRandomMethod(length)
            return s

        self.assertEqual(list(binary.generate(2)), list(reversed(["00", "01", "10", "11"])))
        self.assertListEqual(shared_var, [2, 2, 2, 2])

    def test_gen_single_call_1(self):
        @generator(strategy='randomized')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        self.assertIn(binary.call(2), ["00", "01", "10", "11"])

    def test_gen_single_call_2(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        self.assertEqual(binary.call(2), "00")

    def test_gen_exception_as_continue(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            if s == "10":
                raise ExceptionAsContinue

            return s

        self.assertEqual(list(binary.generate(2)), ["00", "01", "11"])

    def test_gen_recursive_1(self):
        @generator(strategy='dfs')
        def binary(length: int):
            if length == 0:
                return ""

            return Select(["0", "1"]) + binary(length - 1)

        self.assertEqual(list(binary.generate(2)), ["00", "01", "10", "11"])

    def test_gen_recursive_2(self):
        @generator(strategy='dfs', caching=True)
        def binary(length: int):
            if length == 0:
                return ""

            return binary(length - 1) + Select(["0", "1"])

        self.assertEqual(list(binary.generate(2)), ["00", "01", "10", "11"])

    def test_gen_recursive_3(self):
        """ The non tail-recursive nature tests generator-level caching"""

        @generator(strategy='dfs', caching=True)
        def binary(length: int):
            if length == 0:
                return ""

            dummy = binary
            return dummy(length - 1) + Select(["0", "1"])

        self.assertEqual(list(binary.generate(2)), ["00", "01", "10", "11"])

    def test_gen_recursive_4(self):
        @generator(strategy='dfs')
        def binary(length: int):
            if length == 0:
                return ""

            dummy = binary
            return Select(["0", "1"]) + dummy(length - 1)

        self.assertEqual(list(binary.generate(2)), ["00", "01", "10", "11"])

    def test_gen_mutually_recursive_1(self):
        @generator(strategy='dfs')
        def binary1(length: int):
            if length == 0:
                return ""

            return Select(["0", "1"]) + binary2(length - 1)

        @generator(strategy='dfs')
        def binary2(length: int):
            if length == 0:
                return ""

            return Select(["0", "1"]) + binary1(length - 1)

        self.assertEqual(list(binary1.generate(2)), ["00", "01", "10", "11"])
        self.assertEqual(list(binary2.generate(2)), ["00", "01", "10", "11"])

    def test_gen_mutually_recursive_2(self):
        @generator(strategy='dfs')
        def binary(length: int):
            if length == 0:
                return ""

            return Select(["0", "1"]) + binary(length - 1)

        @generator(strategy='dfs')
        def binary1(length: int):
            if length == 0:
                return ""

            return Select(["0", "1"]) + binary2(length - 1)

        @generator(strategy='dfs')
        def binary2(length: int):
            if length == 0:
                return ""

            return Select(["0", "1"]) + binary1(length - 1)

        for l in [2, 3]:
            self.assertEqual(list(binary.generate(l)), list(binary1.generate(l)))
            self.assertEqual(list(binary.generate(l)), list(binary2.generate(l)))

    def test_gen_composition_1(self):
        @generator(group='binary')
        def lower_bit():
            return Select(["0", "1"])

        @generator(group='binary')
        def upper_bit():
            return Select(["0", "1"]) + lower_bit()

        self.assertEqual(list(upper_bit.generate()), ["00", "01", "10", "11"])

    def test_gen_composition_2(self):
        @generator(group='binary')
        def upper_bit():
            return Select(["0", "1"]) + lower_bit()

        @generator(group='binary')
        def lower_bit():
            return Select(["0", "1"])

        self.assertEqual(list(upper_bit.generate()), ["00", "01", "10", "11"])

    def test_gen_composition_without_groups_1(self):
        @generator
        def lower_bit():
            return Select(["0", "1"])

        @generator
        def upper_bit():
            return Select(["0", "1"]) + lower_bit()

        self.assertEqual(list(upper_bit.generate()), ["00", "01", "10", "11"])

    def test_gen_composition_performance_warning(self):
        @generator
        def upper_bit():
            dummy = lower_bit
            return Select(["0", "1"]) + dummy()

        @generator
        def lower_bit():
            return Select(["0", "1"])

        self.assertWarnsRegex(PerformanceWarning,
                              r"Compositional generator invocation discovered at runtime, "
                              r"which may incur a performance penalty.",
                              upper_bit.call)

    def test_gen_composition_with_wrapper(self):
        @generator
        def upper_bit():
            dummy = lower_bit
            return Select(["0", "1"]) + CallGenerator(dummy())

        @generator
        def lower_bit():
            return Select(["0", "1"])

        try:
            success = False
            self.assertWarnsRegex(PerformanceWarning,
                                  r"Compositional generator invocation discovered at runtime, "
                                  r"which may incur a performance penalty.",
                                  upper_bit.call)
        except AssertionError:
            success = True
            pass

        if not success:
            self.fail("Performance warning is erroneously generated.")

    def test_gen_hooks_basic_1(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        self.assertEqual([i[0] for i in list(binary.with_env(tracing=True).generate(2))], ["00", "01", "10", "11"])

    def test_gen_replay_basic_1(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        traces = [i[1] for i in binary.with_env(tracing=True).generate(2)]

        self.assertEqual([list(binary.with_env(replay=t).generate(2))[0] for t in traces], ["00", "01", "10", "11"])
        #  Arguments to generate omitted
        self.assertEqual([list(binary.with_env(replay=t).generate())[0] for t in traces], ["00", "01", "10", "11"])

        #  Using `call` and normal __call__
        self.assertEqual(binary.with_env(replay=traces[0]).call(2), "00")
        self.assertEqual(binary.with_env(replay=traces[0])(2), "00")
        self.assertEqual(binary.with_env(replay=traces[0]).call(), "00")
        self.assertEqual(binary.with_env(replay=traces[0])(), "00")

    def test_gen_replay_randomized_1(self):
        @generator(strategy='randomized')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        values, traces = zip(*[i for i in itertools.islice(binary.with_env(tracing=True).generate(2), 50)])
        self.assertEqual([binary.with_env(replay=t).call(2) for t in traces], list(values))
        #  Arguments to call omitted
        self.assertEqual([binary.with_env(replay=t).call() for t in traces], list(values))

    def test_gen_replay_with_labels(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"], uid="bit_select")

            return s

        self.assertEqual(binary.with_env(replay={"bit_select": ["0", "1"]}).call(2), "01")

    def test_gen_class_method_1(self):
        class TestClass:
            @generator
            def gen_method1(self):
                return Select([1, 2, 3])

        t = TestClass()
        self.assertEqual(list(t.gen_method1.generate()), [1, 2, 3])

    def test_gen_class_method_2(self):
        class TestClass:
            @generator
            def gen_method1(self):
                return self.gen_method2()

            @generator
            def gen_method2(self):
                return Select([1, 2, 3])

        t = TestClass()
        self.assertEqual(list(t.gen_method1.generate()), [1, 2, 3])


class TestGeneratorCompilation(unittest.TestCase):
    def test_arg_handling_1(self):
        @generator
        def binary(length: int, *args, **kwargs):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        self.assertEqual(list(binary.generate(2, 'dummy')), ["00", "01", "10", "11"])


class TestGeneratorModels(unittest.TestCase):
    def test_model_1(self):
        class TestModel(GeneratorModel):
            def save(self, path: str):
                pass

            @classmethod
            def load(cls, path: str):
                pass

            def train(self, data: Any, *args, **kwargs):
                pass

            def infer(self, domain: Any, context: Any = None, op_info: OpInfo = None, **kwargs):
                yield from reversed(domain)

        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        self.assertEqual(list(binary.with_env(model=TestModel()).generate(2)), ["11", "10", "01", "00"])
        self.assertEqual([r for r, t in binary.with_env(model=TestModel(), tracing=True).generate(2)],
                         ["11", "10", "01", "00"])

