from Matrix import Matrix
from Function import Function
import termtables as tt


class Simplex:
    """
    Main simplex class made for calculating the optimal value of input vector for a given function and constraints.

    Attributes
    ----------
    function: Function
        function to optimise

    matrix: Matrix
        matrix of constraints (assumed that all given in the form of inequalities)

    b: Matrix
        right hand side column vector (size n x 1)

    approximation: int | float
        approximation of an answer

    to_maximise: bool
        True - function has to be maximised
        False - function has to be minimised

    Methods
    ----------
    show_table(function, basic, b, matrix, delta_row)
        draws a table with given data in console

    """

    def __init__(self, function: Function, matrix: Matrix, b: Matrix, approximation: int | float,
                 to_maximise: bool = True):
        self.function = function
        self.matrix = matrix
        self.b = b
        self.approximation = approximation

    def optimise(self):
        function = Function(list(self.function.coefficients) + [0] * self.matrix.rows)
        basic = [len(self.function) + i for i in range(self.matrix.rows)]
        b = self.b
        matrix = self.matrix

        for i in range(self.matrix.rows):
            col = [0.0] * matrix.rows
            col[i] = 1.0
            matrix.add_col(col)

        ci = Function([function[basic[i]] for i in range(len(basic))])
        delta_row = ['-', 'delta', ci(b)] + [ci(col) - function[i] for i, col in enumerate(matrix.get_cols())]

        self.show_table(function, basic, b, matrix, delta_row)

    def show_table(self, function, basic, b, matrix, delta_row):
        data = [[function[basic[row]], basic[row], b[row], *[el for el in matrix[row]]] for row in
                range(matrix.rows)] + [delta_row]

        view = tt.to_string(
            data,
            header=['c', 'basic', 'b'] + [f'x{i}' for i in range(len(function))],
            style=tt.styles.rounded_thick
        )
        print(function)
        print(view)


if __name__ == '__main__':
    from parser import parse_file

    s = Simplex(*parse_file('tests/test1.txt'))

    s.optimise()
