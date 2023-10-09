import numpy as np

from Function import Function
from Simplex import Simplex, InfeasibleSolution
from input_parser import parse_file


def run_simplex_from_file(input_file):
    simplex = Simplex(*parse_file(input_file))
    np.set_printoptions(precision=simplex.eps)
    print(simplex)
    try:
        solution = simplex.optimise()
        print(solution)
    except InfeasibleSolution:
        print("SOLUTION:\nThe method is not applicable!")


def get_input_from_console():
    print("Enter the coefficients of the variables (separated by space):")
    C = list(map(float, input().split()))

    print("Enter the coefficients of the constraints. Enter 'done' when finished:")
    A = []
    while True:
        line = input()
        if line == 'done':
            break
        A.append(list(map(float, line.split())))

    print("Enter the right-hand side values of the constraints:")
    b = list(map(float, input().split()))

    print("Enter the number of digits after the decimal point (epsilon):")
    eps = int(input())

    return Function(C), np.array(A), np.array(b), eps


# if __name__ == "__main__":
#     run_simplex_from_file("inputs/input5.txt")
#
if __name__ == "__main__":
    print("Enter input from console")
    C, A, b, eps = get_input_from_console()
    simplex = Simplex(C, A, b, eps)
    np.set_printoptions(precision=eps)
    print(simplex)
    try:
        solution = simplex.optimise()
        print(solution)
    except InfeasibleSolution:
        print("SOLUTION:\nThe method is not applicable!")
