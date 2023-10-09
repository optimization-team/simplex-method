import numpy as np
from input_parser import parse_file
from Simplex import Simplex, InfeasibleSolution


def run_simplex_from_file(input_file):
    simplex = Simplex(*parse_file(input_file))
    np.set_printoptions(precision=simplex.eps)
    print(simplex)
    try:
        solution = simplex.optimise()
        print(solution)
    except InfeasibleSolution:
        print("SOLUTION:\nThe method is not applicable!")


if __name__ == "__main__":
    run_simplex_from_file("inputs/input2.txt")
