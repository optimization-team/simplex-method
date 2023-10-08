"""
Parser for input files and test files.
Parses files into arguments for Simplex and SimplexTestCase classes.

See Also:
    Simplex from Simplex3.py
    SimplexTestCase from test_simplex3.py
"""

from Function import Function

from numpy import matrix, array


def parse_file(filename: str):
    with open(filename) as file:
        function = Function(list(map(float, file.readline().split())))
        file.readline()

        m = list()
        constraint = file.readline()
        while constraint != '\n':
            m.append(list(map(float, constraint.split())))
            constraint = file.readline()
        m = matrix(m)

        b = list(map(float, file.readline().split()))
        file.readline()
        b = array(b)

        approximation = int(file.readline().strip())

        return function, m, b, approximation


def parse_test(filename: str):
    """
    Parse test file with optimal value and optimal vector

    Parameters
    ----------
    filename: str
        name of the file to parse
    """
    # parse file + parse a vector of optimal values and optimal function value after -----
    with open(filename) as file:
        function = Function(list(map(float, file.readline().split())))
        file.readline()

        m = list()
        constraint = file.readline()
        while constraint != '\n':
            m.append(list(map(float, constraint.split())))
            constraint = file.readline()
        m = array(m)

        b = list(map(float, file.readline().split()))
        file.readline()
        b = array(b)

        approximation = int(file.readline().strip())

        for _ in range(3):
            file.readline()

        fun = float(file.readline().strip())
        file.readline()
        x = list(map(float, file.readline().split()))

        return function, m, b, approximation, x, fun


if __name__ == '__main__':
    f, m, b, a, x, opt = parse_test('tests/test1.txt')
    # f, m, b, a = parse_file('inputs/input1.txt')
    print(f)
    print()

    print(m)
    print()

    print(b)
    print()

    print(a)
    print()

    print(x)
    print()

    print(opt)
    print()
