"""
Module for Function class.
Function class is used to store a linear function by given coefficients.
"""
from __future__ import annotations


class Function:
    """
    Function class

    Attributes
    ----------
    coefficients: list[int | float]
        list of coefficients of a function

    Methods
    ----------
    __init__(*args, coefficients):
        Takes as an argument list of values (vector) or a sequence of numbers, not both.

    __call__(*args, vector) -> float | int:
        Takes as an argument list of values (vector) or a sequence of numbers, not both.
        Then returns a dot product of coefficients and a given data.
    """

    def __init__(self, *args, coefficients: list[int | float] = None):
        if args and coefficients:
            raise ValueError("Pass vector or list of values, not both")

        if len(args) == 1 and (isinstance(args[0], list) or isinstance(args[0], tuple)):
            args = tuple(args[0])

        if coefficients:
            args = tuple(coefficients)

        self.coefficients = args

    def __call__(self, *args, vector=None) -> int | float:
        if args and vector:
            raise ValueError("Pass vector or list of values, not both")

        if len(args) == 1 and (isinstance(args[0], list) or isinstance(args[0], tuple)):
            args = tuple(args[0])

        if vector:
            args = tuple(vector)
        if len(args) != len(self.coefficients):
            raise ValueError("Input size does not match the size of the function")

        val = 0
        for c, v in zip(self.coefficients, args):
            val += c * v

        return val

    def __str__(self):
        return f'Function({", ".join(map(str, self.coefficients))})'

    def __len__(self):
        return len(self.coefficients)

    def __getitem__(self, item: int):
        return self.coefficients[item]

    def invert_sign(self):
        self.coefficients = [-c for c in self.coefficients]


if __name__ == '__main__':
    f = Function(1, 2, 3)  # also possible Function((1, 2, 3)) and Function([1, 2, 3])
    res = f(3, 2, 1)  # also possible f((3, 2, 1)) and f([3, 2, 1])
    print(res)
    f.invert_sign()
    res = f(3, 2, 1)
    print(res)
