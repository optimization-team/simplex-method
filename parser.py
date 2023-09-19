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


if __name__ == '__main__':
    f, m, b, a = parse_file('tests/test1.txt')
    print(f)
    print()

    print(m)
    print()

    print(b)
    print()

    print(a)
    print()
