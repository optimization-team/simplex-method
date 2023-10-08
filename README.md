# Programing task 1, "Introduction to optimization"

## How to test the program on your LPP
TODO: Change if changes
To test the program on your input, change the contents of the file ["input1.txt"](https://github.com/optimization-team/simplex-method/blob/main/inputs/input1.txt) on the information about your input.

The input contains:
- A vector of coefficients of objective function - C.
- A matrix of coefficients of constraint function - A.
- A vector of right-hand side numbers - b.
- The approximation accuracy ϵ.

Then, run the Simplex.py file, and check if the output you get is what you expected.
## Structure of the project
### [inputs](https://github.com/optimization-team/simplex-method/tree/main/inputs)
Folder, containing 5 different inputs, on which the program was tested.
### [tests](https://github.com/optimization-team/simplex-method/tree/main/tests)
Folder, containing 5 different inputs and correct answers for those inputs, on which the program was tested.
### [Function.py](https://github.com/optimization-team/simplex-method/blob/main/Function.py)
File containing a Function class. This class is used to store the objective function for the LPP.
### [Simplex.py](https://github.com/optimization-team/simplex-method/blob/main/Simplex.py)
File containing the Simplex method itself. Contains the following classes:
- SimplexSolution - class, used to store the solution for the LPP.
- InfeasibleSolution - exception, thrown when there is no feasible solution.
- Simplex - class, responsible for calculating the LPP using the Simplex method.
### [input_parser.py](https://github.com/optimization-team/simplex-method/blob/main/input_parser.py)
File containing functions parsing input into format, needed for the Simplex class.
### [main.py](https://github.com/optimization-team/simplex-method/blob/main/main.py)
TODO: Add info about main.py.
### [requirements.py](https://github.com/optimization-team/simplex-method/blob/main/requirements.txt)
Information about assets needed for the program to be executed correctly.
### [test_simplex.py](https://github.com/optimization-team/simplex-method/blob/main/test_simplex.py)
File containing the classes and functions needed to test the program on the tests, given in "tests" folder.
