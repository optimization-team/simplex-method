from Matrix import Matrix
from Function import Function


def parse_file(filename: str):
    with open(filename) as file:
        function = Function(list(map(float, file.readline().split())))
        file.readline()

        m = Matrix(rows=0, cols=len(function))
        constraint = file.readline()
        while constraint != '\n':
            m.add_row(list(map(float, constraint.split())))
            constraint = file.readline()

        b = list(map(float, file.readline().split()))
        file.readline()

        approximation = float(file.readline().strip())

        return function, m, b, approximation


def parse_test(filename: str):
    """
    Parse test file with optimal value and optimal vector
    """
    # parse file + parse a vector of optimal values and optimal function value after -----
    with open(filename) as file:
        function = Function(list(map(float, file.readline().split())))
        file.readline()

        m = Matrix(rows=0, cols=len(function))
        constraint = file.readline()
        while constraint != '\n':
            m.add_row(list(map(float, constraint.split())))
            constraint = file.readline()

        b = list(map(float, file.readline().split()))
        file.readline()

        approximation = int(file.readline().strip())
        for _ in range(3): file.readline()
        fun = float(file.readline().strip())
        file.readline()
        x = list(map(float, file.readline().split()))

        return function, m, b, approximation, x, fun


if __name__ == '__main__':
    f, m, b, a = parse_file('inputs/input1.txt')
    print(f)
    print()

    print(m)
    print()

    print(b)
    print()

    print(a)
    print()
