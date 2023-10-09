# Programing task 1, "Introduction to optimization"

## How to test the program on your LPP
To test the program on your input, run the main.py file.
- Choose, if you want to input the LPP from console or from file.
- Choose, if your LPP is maximization or minimization.
- Choose, if you want intermidiate steps of solution to be printed.
- If you chose input from console
  - Enter objective value coefficients, separated by space, in one line.
  - Enter coefficients of constraints, line by line. Print "done" when finished.
  - Enter right-hand side values of constraints.
  - Enter number of digits after decimal point (precision).
- If you chose input from file, specify the name of the file from [inputs](https://github.com/optimization-team/simplex-method/tree/main/inputs) folder you want to use
- Check, if the results are what you expected.
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
File, from which the program can be tested on the input from console or from certain file from [inputs](https://github.com/optimization-team/simplex-method/tree/main/inputs) folder.
### [requirements.py](https://github.com/optimization-team/simplex-method/blob/main/requirements.txt)
Information about assets needed for the program to be executed correctly.
### [test_simplex.py](https://github.com/optimization-team/simplex-method/blob/main/test_simplex.py)
File containing the classes and functions needed to test the program on the tests, given in "tests" folder.
