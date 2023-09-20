class Matrix:
    """
    Matrix class

    Attributes
    ----------
    rows: int
        number of rows
    cols: int
        number of columns
    data: list[list[int]]
        matrix data itself

    Methods
    ----------
    add_row(row: list[float|int])
        Adds a new row passed as an argument to a matrix
    """

    def __init__(self, rows, cols):
        """
        Constructor

        :param rows: initial number of rows in a matrix
        :param cols: initial number of columns in a matrix
        """
        self.rows = rows
        self.cols = cols
        self.data = [[0 for _ in range(cols)] for __ in range(rows)]

    def add_row(self, row: list[float | int]) -> None:
        """
        Adds a row to the bottom of a matrix

        :param row: row to be added
        :return: None
        """
        if not isinstance(row, list):
            raise ValueError("Undefined behaviour")
        if len(row) != self.cols:
            raise ValueError("Wrong length of the row")
        self.data.append(row)
        self.rows += 1

    def add_col(self, col: list[float | int]) -> None:
        """
        Adds a col to the right of a matrix

        :param col: col to be added
        :return: None
        """
        if not isinstance(col, list):
            raise ValueError("Undefined behaviour")
        if len(col) != self.rows:
            raise ValueError("Wrong length of the row")
        for row in range(self.rows):
            self.data[row].append(col[row])
        self.cols += 1

    def get_cols(self):
        cols = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return cols

    def __str__(self) -> str:
        return '[' + '\n'.join(['[' + '\t'.join(map(str, row)) + ']' for row in self.data]) + ']'

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, key) -> list[float | int]:
        return self.data[key]

    def __setitem__(self, key, value) -> None:
        if isinstance(value, Matrix):
            if value.rows == self.rows and value.cols == self.cols:
                self.data[key] = value.data
            else:
                raise ValueError("Matrix dimensions must match for assignment.")
        elif isinstance(value, list) and len(value) == self.cols:
            self.data[key] = value
        else:
            raise ValueError("Undefined behaviour")

    def __iadd__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition.")

        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] += other.data[i][j]
        return self

    def __isub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for subtraction.")

        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] -= other.data[i][j]
        return self

    def __imul__(self, other):
        if self.cols != other.rows:
            raise ValueError(
                "Number of columns in the first matrix must match the number of rows in the second matrix for multiplication.")

        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]

        self.data = result.data
        self.cols = other.cols
        return self

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition.")

        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for subtraction.")

        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] = self.data[i][j] - other.data[i][j]
        return result

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(
                    "Number of columns in the first matrix must match the number of rows in the second matrix for multiplication.")

            result = Matrix(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    for k in range(self.cols):
                        result[i][j] += self.data[i][k] * other.data[k][j]
            return result
        elif isinstance(other, (int, float)):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i][j] = self.data[i][j] * other
            return result
        else:
            raise ValueError("Unsupported operand type for multiplication.")


if __name__ == '__main__':
    help(Matrix)

    a = Matrix(2, 3)
    a[0] = [1, 2, 3]
    a[1] = [4, 5, 6]

    b = Matrix(3, 2)
    b[0] = [1, 2]
    b[1] = [3, 4]
    b[2] = [5, 6]

    print(a * b)

    print(a.get_cols())
