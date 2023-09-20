# run from cmd with: pytest
import pytest
import os
from Simplex import Simplex
from Function import Function
from Matrix import Matrix
from parser import parse_test


class SimplexTestCase:
    """
    Class for storing test cases

    Attributes
    ----------
    simplex: Simplex
        Simplex object
    x: list[float]
        list of optimal values
    opt: int | float
        optimal function value
    """

    def __init__(self, function: Function, matrix: Matrix, b: Matrix, approximation: int | float,
                 x: list[float], opt: int | float):
        self.simplex = Simplex(function, matrix, b, approximation)
        self.x = x
        self.opt = opt

    def __str__(self):
        return f'TestCase:\n{self.simplex.function},\n' \
               f'A:\n{self.simplex.matrix},\n' \
               f'b: {self.simplex.b},\n' \
               f'accuracy: {self.simplex.approximation},\n' \
               f'Vector of decision variables: ({", ".join(map(str, self.x))}),\n' \
               f'Optimal value of objection function: {self.opt}'


class TestSimplex:
    """
    Class for testing Simplex class

    Attributes
    ----------
    tests: list[str]
        list of test files
    test_cases: list[str]
        list of test cases

    Methods
    ----------
    test_simplex(test_file)
        test simplex method on a test case with parametrization on list of test cases
    """
    tests = os.listdir('tests')
    test_cases = [os.path.join('tests', file) for file in tests]

    @pytest.mark.parametrize("test_file", test_cases)
    def test_simplex(self, test_file):
        testcase = SimplexTestCase(*parse_test(test_file))

        # opt, x = testcase.simplex.optimize()
        opt, x = testcase.simplex.plug_optimize()

        for i in range(len(x)):
            assert round(x[i], testcase.simplex.approximation) == \
                   round(testcase.x[i], testcase.simplex.approximation)

        assert opt == testcase.opt


