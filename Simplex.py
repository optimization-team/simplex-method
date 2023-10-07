"""
Module for solving maximization and minimization problems using simplex method.
"""
from __future__ import annotations

from enum import Enum
from Function import Function
import termtables as tt
from scipy.optimize import linprog

import numpy as np


class Solution(Exception):
    def __init__(self, X, z):
        self.X = X
        self.z = z


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

    def compute_basic_solution(self, B_inv) -> None:
        x_B = np.dot(B_inv, self.b)
        for i in range(len(self.basic_indices)):
            self.x[self.basic_indices[i] - 1] = x_B[i - 1]

    def update_basis(self) -> None:

        class MinMax(Enum):
            MIN = 1
            MAX = 2

            @staticmethod
            def from_to_minimize():
                if self.to_minimize:
                    return MinMax.MIN
                else:
                    return MinMax.MAX

        def get_non_basis(A, basis):
            variables = np.arange(len(A[0]))
            mask = np.ones(variables.size, dtype=bool)
            mask[basis] = False
            return variables[mask]

        def is_optimal(self, B_inv) -> bool:
            # optimal if z-c >= 0
            # z = c_B * B_inv * A' ,
            difference = self.c_B @ B_inv @ get_non_basis(self.A, self.basic_indices) - self.c
            if self.to_minimize:
                for i in difference:
                    if i > 0:
                        return False
            else:
                for i in difference:
                    if i < 0:
                        return False
            return True

        # basis is number of
        def new_basis(B, A, b, basis, C, C_basis, minMax):
            Xb = np.matmul(np.linalg.inv(B), b)
            non_basis = get_non_basis(A, basis)
            optimality = np.matmul(np.matmul(C_basis, np.linalg.inv(B)), A[:, non_basis]) - C[non_basis]
            if minMax == MinMax.MIN:
                if (optimality <= 0).all():
                    raise Solution(X=Xb, z=np.matmul(C_basis, Xb))
                entering_vec_pos = non_basis[np.where(optimality == np.max(optimality))[0][0]]
            elif minMax == MinMax.MAX:
                if (optimality >= 0).all():
                    raise Solution(X=Xb, z=np.matmul(C_basis, Xb))
                entering_vec_pos = non_basis[np.where(optimality == np.min(optimality))[0][0]]
            entering_vec = np.matmul(np.linalg.inv(B), A[:, entering_vec_pos])
            if (entering_vec <= 0).all():
                raise Exception("Unbounded solution")
            np.seterr(divide='ignore')
            feasibility = np.divide(Xb, entering_vec)
            feasibility = feasibility[feasibility != np.inf]
            feasibility = feasibility[feasibility > 0]
            if (feasibility <= 0).all():
                raise Exception("Infeasible")
            leaving_var = np.where(feasibility == np.min(feasibility))[0][0]
            B[:, leaving_var] = entering_vec
            basis[leaving_var] = entering_vec_pos
            C_basis[leaving_var] = C[entering_vec_pos]
            return B, basis, C_basis

        (B, basis, C_basis) = new_basis(self.B, self.A, self.b, self.basic_indices, self.c, self.c_B,
                                        MinMax.from_to_minimize())
        self.B = B
        self.basic_indices = basis
        self.c_B = C_basis

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
        """
        Plug method for autotests
        Returns
        -------
        (int | float, list[int | float])
            optimal value and optimal vector

        """
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
    from input_parser import parse_file, parse_test

    s = Simplex(*parse_file('inputs/input2.txt'))

    opt, x = s.plug_optimize()

    print(opt)
    print(x)
