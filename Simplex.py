from Matrix import Matrix
from Function import Function


class Simplex:
    def __init__(self, function: Function, matrix: Matrix, b: Matrix, approximation: int | float):
        self.function = function
        self.matrix = matrix
        self.b = b
        self.approximation = approximation


