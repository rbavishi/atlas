import itertools
import unittest

from atlas import generator
from atlas.operators import operator
from atlas.strategies import DfsStrategy
from atlas.utils.stubs import stub


@stub
def Select(*args, **kwargs):
    pass


class TestBasicStrategyFunctionality(unittest.TestCase):
    def test_operator_recognition_1(self):
        class TestStrategy(DfsStrategy):
            @operator
            def MyOperator(self):
                pass

        s = TestStrategy()
        self.assertIn('MyOperator', s.known_ops)

    def test_operator_recognition_2(self):
        class TestStrategy(DfsStrategy):
            @operator(name='AnotherName')
            def MyOperator(self):
                pass

        s = TestStrategy()
        self.assertIn('AnotherName', s.known_ops)
        self.assertNotIn('MyOperator', s.known_ops)

    def test_operator_resolution_1(self):
        class TestStrategy(DfsStrategy):
            @operator(name='Select', uid="10")
            def MyOperator(self, domain, **kwargs):
                yield from reversed(domain)

        @generator(strategy=TestStrategy())
        def binary(l: int):
            s = ""
            for i in range(l):
                s += Select(["0", "1"], uid="10")

            return s

        self.assertEqual(list(binary.generate(2)), ["11", "10", "01", "00"])

    def test_operator_resolution_2(self):
        class TestStrategy(DfsStrategy):
            @operator(name='Select', uid="10")
            def MyOperator1(self, domain, **kwargs):
                yield from reversed(domain)

            @operator(name='Select', gen_name="something_else")
            def MyOperator2(self, domain, **kwargs):
                yield from reversed(domain)

        @generator(strategy=TestStrategy())
        def binary(l: int):
            s = ""
            for i in range(l):
                s += Select(["0", "1"], uid="10")

            return s

        self.assertEqual(list(binary.generate(2)), ["11", "10", "01", "00"])

    def test_operator_resolution_3(self):
        class TestStrategy(DfsStrategy):
            @operator(name='Select', tags=["10"])
            def MyOperator1(self, domain, **kwargs):
                yield from reversed(domain)

            @operator(name='Select', gen_name="something_else")
            def MyOperator2(self, domain, **kwargs):
                yield from reversed(domain)

        @generator(strategy=TestStrategy())
        def binary(l: int):
            s = ""
            for i in range(l):
                s += Select(["0", "1"], tags=["10"])

            return s

        self.assertEqual(list(binary.generate(2)), ["11", "10", "01", "00"])

    def test_operator_resolution_4(self):
        class TestStrategy(DfsStrategy):
            @operator(name='Select', uid="10")
            def MyOperator1(self, domain, **kwargs):
                yield from reversed(domain)

            @operator(name='Select', gen_name="binary")
            def MyOperator2(self, domain, **kwargs):
                yield from reversed(domain)

        @generator(strategy=TestStrategy())
        def binary(l: int):
            s = ""
            for i in range(l):
                s += Select(["0", "1"], uid="10")

            return s

        self.assertRaisesRegex(ValueError, r"Could not resolve \.*", lambda x: list(binary.generate(x)), 2)

    def test_operator_resolution_5(self):
        class TestStrategy(DfsStrategy):
            @operator(name='MyOperator', uid="something")
            def MyOperator(self, domain, **kwargs):
                yield from reversed(domain)

        @generator(strategy=TestStrategy())
        def binary(l: int):
            s = ""
            for i in range(l):
                s += MyOperator(["0", "1"], uid="else")

            return s

        self.assertRaisesRegex(ValueError, r"Could not resolve \.*", lambda x: list(binary.generate(x)), 2)

    def test_operator_resolution_6(self):
        class TestStrategy(DfsStrategy):
            @operator(name='Select', tags=["20", "10"])
            def MyOperator1(self, domain, **kwargs):
                yield from reversed(domain)

            @operator(name='Select', tags=["10"])
            def MyOperator2(self, domain, **kwargs):
                yield from reversed(domain)

        @generator(strategy=TestStrategy())
        def binary(l: int):
            s = ""
            for i in range(l):
                s += Select(["0", "1"], tags=["10"])

            return s

        self.assertRaisesRegex(ValueError, r"Could not resolve \.*", lambda x: list(binary.generate(x)), 2)

    def test_randomized_operators(self):
        @generator(strategy='randomized')
        def all_ops():
            ctx = {"abc": 123, "def": 456}

            domain = list(range(5))

            a = Select(domain, context=ctx)
            b = Sequence(domain, context=ctx, max_len=4)
            c = Subset(domain, context=ctx)
            d = OrderedSubset(domain, context=ctx, lengths=[2, 3])

            return a, b, c, d

        domain = list(range(5))
        for res in itertools.islice(all_ops.generate(), 100):
            self.assertIn(res[0], domain)
            for elem in res[1]:
                self.assertIn(elem, domain)

            self.assertEqual(len(res[2]), len(set(res[2])))
            self.assertEqual(len(res[3]), len(set(res[3])))
            self.assertIn(len(res[3]), [2, 3])
