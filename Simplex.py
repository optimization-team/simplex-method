from __future__ import annotations

from Function import Function
import termtables as tt
from scipy.optimize import linprog

import numpy as np


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

    plug_optimize()
        plug method for autotests
        Returns optimal value and optimal vector

    """

    def __init__(self, function: Function, A: np.array, b: np.array, approximation: int,
                 to_minimize: bool = False):
        np.set_printoptions(precision=approximation)

        # constants
        self.function = function
        self.A = A
        self.b = b
        self.to_minimize = to_minimize
        n, m = np.shape(self.A)  # number of vars and number of constraints

        # variables
        self.x = np.zeros((n, 1))  # current solution, where x_B = [x[i] for i in basic_indices]
        self.basic_indices = [i for i in range(n - m + 1, n + 1)]  # each index is in range 1,...,n
        self.B = np.identity(len(self.b))
        self.c_B = np.zeros((1, len(self.basic_indices)))  # basic var coefs
        self.c = self.function.coefficients  # nonbasic var coefs

        # min or max?
        if to_minimize:
            self.function.invert_sign()

    def is_optimal(self, B_inv) -> bool:
        # optimal if z-c >= 0
        # z = c_B * B_inv * A' ,
        # where A' - all nonbasic cols of A
        pass

    def compute_basic_solution(self, B_inv) -> None:
        x_B = np.dot(B_inv, self.b)
        for i in range(len(self.basic_indices)):
            self.x[self.basic_indices[i] - 1] = x_B[i - 1]


    def update_basis(self) -> None:
        # determines the leaving and entering vars and updates the fields: basic_indices, B, c, and c_B
        B_inv = np.linalg.inv(self.B)  # Step 1: Compute the inverse of the basis B

        # Step 2: Calculate z - c for non-basic variables
        z_minus_c = np.dot(self.c_B, np.dot(B_inv, self.A) - self.c)

        if all(z_minus_c >= 0):  # If z - c >= 0 for all non-basic vectors, we have reached the optimal solution
            return

        entering_var_index = np.argmin(z_minus_c)  # Find entering vector

        A_i = np.dot(B_inv, self.A[:, entering_var_index])  # Step 3: Compute B_inv * A_i

        # Step 3 (continued): Determine the leaving vector using the ratio test
        ratios = np.divide(self.x[self.basic_indices - 1], A_i)
        ratios[A_i <= 0] = np.inf  # Ignore negative or zero elements in A_i
        leaving_var_index = np.argmin(ratios)

        # Step 4: Update the basis
        leaving_var = self.basic_indices[leaving_var_index]
        self.basic_indices[leaving_var_index] = entering_var_index + 1

        E = np.eye(len(self.b))
        E[:, leaving_var_index] = -A_i / A_i[leaving_var_index]
        self.B = np.dot(E, self.B)

        # Update c_B and c
        self.c_B = self.function.coefficients[self.basic_indices - 1]
        self.c = self.function.coefficients
        self.c[self.basic_indices - 1] = 0

        # Update x
        self.x[leaving_var - 1] = 0

        pass

    def optimise(self) -> None:
        self.compute_basic_solution(self.B)
        B_inv = self.B
        while not self.is_optimal(B_inv):
            self.update_basis()
            self.compute_basic_solution(np.invert(self.B))

        if self.to_minimize:
            print("The min value is achieved at " + str(self.x) + ", and it is " + str(-self.function(self.x)))
        else:
            print("The max value is achieved at " + str(self.x) + ", and it is " + str(self.function(self.x)))

    def plug_optimize(self) -> (int | float, list[int | float]):
        function = Function(list(self.function.coefficients))
        b = self.b
        matrix = self.A
        # change sign of function coefficients,
        # because linprog solves the minimization problem
        function.coefficients = [-el for el in function.coefficients]
        opt = linprog(
            c=function.coefficients,
            A_ub=matrix,
            b_ub=b,
            A_eq=None,
            b_eq=None,
            bounds=[(0, float("inf")) for _ in range(len(function))],
            method="highs"
        )
        return -opt.fun, opt.x


if __name__ == '__main__':
    from parser import parse_file, parse_test

    s = Simplex(*parse_file('inputs/input2.txt'))

    opt, x = s.plug_optimize()

    print(opt)
    print(x)
