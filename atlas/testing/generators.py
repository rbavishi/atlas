import unittest

from atlas import generator
from atlas.strategies import DfsStrategy


def Select(*args, **kwargs):
    pass


class GeneratorBasic(unittest.TestCase):
    def test_gen_single_1(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        self.assertEqual(list(binary.generate(2)), ["00", "01", "10", "11"])

    def test_gen_custom_strategy_1(self):
        class ReversedDFS(DfsStrategy):
            def Select_reversed(self, domain, *args, **kwargs):
                yield from reversed(domain)

        @generator(strategy=ReversedDFS())
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"], label='reversed')

            return s

        self.assertEqual(list(binary.generate(2)), list(reversed(["00", "01", "10", "11"])))

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

    def test_gen_hooks_basic_1(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        self.assertEqual([i[0] for i in list(binary.generate(2).with_tracing())], ["00", "01", "10", "11"])

    def test_gen_replay_basic_1(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        traces = [i[1] for i in binary.generate(2).with_tracing()]

        self.assertEqual([list(binary.generate(2).replay(t))[0] for t in traces], ["00", "01", "10", "11"])

    def test_gen_replay_basic_2(self):
        @generator(strategy='dfs')
        def binary(length: int):
            s = ""
            for i in range(length):
                s += Select(["0", "1"])

            return s

        traces = [i[1] for i in binary.generate(2).with_tracing()]

        #  Arguments to generate omitted
        self.assertEqual([list(binary.generate().replay(t))[0] for t in traces], ["00", "01", "10", "11"])
