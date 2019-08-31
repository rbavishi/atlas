import unittest
from typing import Any, List

import pandas as pd
import numpy as np

import atlas.synthesis.numpy.api
from atlas import generator
from atlas.exceptions import ExceptionAsContinue
from atlas.utils import get_group_by_name

api_gens = {
    gen.name: gen for gen in get_group_by_name('numpy')
}


@generator(group='numpy')
def simple_enumerator(inputs, output, func_seq):
    prog = []
    intermediates = []
    for func in func_seq:
        func = api_gens[func]
        try:
            val, args = func(intermediates + inputs, output)
        except Exception as e:
            raise ExceptionAsContinue
        prog.append((func.name, args))
        intermediates.append(val)

    return intermediates[-1], prog


class TestGenerators(unittest.TestCase):
    def check(self, inputs: List[Any], output: Any, funcs: List[str], seqs: List[List[int]],
              constants: List[Any] = None):
        if constants is not None:
            inputs += constants

        def checker(v1, v2):
            if not isinstance(v2, np.ndarray):
                return False
            return np.array_equal(v1, v2)

        func_seqs = [[funcs[i] for i in seq] for seq in seqs]
        for func_seq in func_seqs:
            for val, prog in simple_enumerator.generate(inputs, output, func_seq):
                if checker(output, val):
                    return True

        self.assertTrue(False, "Did not find a solution")

    def test_ndarray_reshape(self):
        inputs = [np.array(range(28))]
        output = inputs[0].reshape([4, 7])
        funcs = ['ndarray.reshape']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_ndarray_reshape_add_dim(self):
        inputs = [np.array(range(28))]
        output = inputs[0].reshape([2, 1, 2, 7])
        funcs = ['ndarray.reshape']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_ndarray_flatten(self):
        inputs = [np.array([[0, 1], [3, 4]])]
        output = inputs[0].flatten()
        funcs = ['ndarray.flatten']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_ndarray_transpose(self):
        inputs = [np.array([[0, 1], [3, 4]])]
        output = inputs[0].transpose()
        funcs = ['ndarray.transpose']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_ndarray_transpose_complex(self):
        inputs = [np.array(range(28)).reshape([2, 2, 7])]
        output = inputs[0].transpose(0, 2, 1)
        funcs = ['ndarray.transpose']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)
