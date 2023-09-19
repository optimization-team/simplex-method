class Function:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __call__(self, *args, vector):
        if args and vector:
            raise ValueError("Pass vector or list of values, not both")

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
