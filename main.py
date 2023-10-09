import numpy as np
from Function import Function
from Simplex import Simplex, InfeasibleSolution
from input_parser import parse_file


def read_input_from_console() -> (Function, np.ndarray, np.ndarray, int):
    """
    Reads the input from the console.
    """
    print("Enter the objective function coefficients (separated by space):")
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


def read_input_from_file(input_file) -> (Function, np.ndarray, np.ndarray, int):
    """
    Reads the input from the given file.
    """
    if not input_file.endswith(".txt"):
        print("File is not a txt file. Please choose another file.")
        return None
    try:
        return parse_file("inputs/" + input_file)
    except FileNotFoundError:
        print("File not found in \"inputs\" directory. Please choose another file.")
        return None


def run_simplex(C, A, b, eps, max_min, print_steps) -> None:
    """
    Runs the simplex algorithm on the given input.
    """
    np.set_printoptions(precision=eps)
    simplex = Simplex(C, A, b, eps, max_min == "max")
    print(simplex)
    try:
        solution = simplex.optimise(print_iterations=print_steps)
        print(solution)
    except InfeasibleSolution:
        print("SOLUTION:\nThe method is not applicable!")


if __name__ == "__main__":
    print("Do you want to input from the console or from a file? (console/file)")
    choice = input().strip().lower()

    print("Do you want to maximize or minimize? (max/min)")
    max_min = input().strip().lower()

    # print("Should the iteration steps be printed? (y/n)")
    # print_steps = input().strip().lower() == "y"

    if choice == "console":
        print("Enter input from console")
        C, A, b, eps = read_input_from_console()
        run_simplex(C, A, b, eps, max_min, print_steps=False)
    elif choice == "file":
        print("Enter the name of the input file from the \"inputs\" folder (e.g., 'input1.txt'):")
        input_file = None
        while input_file is None:
            input_file = read_input_from_file(input().strip())
        C, A, b, eps = input_file
        run_simplex(C, A, b, eps, max_min, print_steps=False)
    else:
        print("Invalid choice. Please choose 'console' or 'file'.")
