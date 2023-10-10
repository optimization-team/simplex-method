from dataclasses import dataclass
import numpy as np
import termtables as tt
from Function import Function


@dataclass
class SimplexSolution:
    """
    Custom exception class for the Simplex algorithm. Contains solution for an optimization problem.

    Attributes
    ----------
    x: np.array
    opt: float
    """

    x: np.array
    opt: float

    def __str__(self):
        return (
            f"SOLUTION: \n"
            f"Vector of decision variables: ({', '.join(map(str, self.x))}),\n"
            f"Optimal value: {self.opt}"
        )


class InfeasibleSolution(Exception):
    def __init__(self):
        super().__init__("Infeasible solution, method is not applicable!")


class AlternatingOptima(Exception):
    def __init__(self, solution):
        super().__init__("Alternating optima detected!")
        self.solution = solution


class Simplex:
    """
    Main simplex class made for calculating the optimal value of input vector for a given function and constraints.

    Attributes
    ----------
        Initial variables:
        ------------------
        function: Function

        constraints_matrix: np.array

        C: np.array
            function with slack variables

        A: np.array
            matrix of constraints (assumed that all given in the form of inequalities) with slack variables

        b: np.array
            right hand side column vector (size n x 1)

        eps: int
            approximation of an answer (number of digits after comma)

        m: int
            number of constraints

        n: int
            number of variables

        Simplex table variables:
        ------------------------
        B: np.array
            basis matrix

        C_B: np.array
            vector of coefficients of the basis variables

        basic: list[int]
            list of indices of basic variables

        X_B: np.array
            vector of basic variables

        z: float
            value of the objective function

        C_B_times_B_inv: np.array
            helper vector for calculating z
    """

    def __init__(
            self,
            C: Function,
            A: np.array,
            b: np.array,
            eps: int = 2,
            to_maximize=True,
    ):

        assert A.ndim == 2, "A is not a matrix"
        assert b.ndim == 1, "b is not a vector"
        assert (A.shape[0] == b.size), \
            "Length of right-handside vector of the constraints does not correspond to the number of rows of matrix A"
        assert A.shape[1] == len(C), \
            "Number of variables in objective function does not correspond to the number of cols of matrix A"

        self.to_maximize = to_maximize
        self.function = C
        self.constraints_matrix = A

        self.C = np.hstack((np.array(C.coefficients), np.zeros(A.shape[0])))
        self.A = np.hstack((A, np.identity(A.shape[0])))

        self.b = b
        self.eps = eps
        self.epsilon = 1 / (10 ** self.eps)
        self.m, self.n = self.A.shape
        self.z_row = []
        self.B = np.identity(self.m)
        self.C_B = np.zeros(self.m)
        self.basic = list(range(self.n - self.m, self.n))
        self.X_B = np.zeros(self.m)
        self.z = 0.0
        self.delta = 0.0
        self.C_B_mul_B_inv = np.zeros(self.m)
        self.iteration = 0

    def __str__(self):
        to_maximize = "max" if self.to_maximize else "min"
        constraints = f""
        for i in range(self.m):
            constraints += f"{self.constraints_matrix[i]} <= {self.b[i]}\n"
        constraints = constraints[:-1]
        constraints = constraints.replace("[[", "|").replace("]]", "|")
        approximation = f"Approximation: {self.eps}"
        return f"LPP:\n{self.function} -> {to_maximize}\n{constraints}\n{approximation}\n"

    def optimise(self, print_iterations) -> SimplexSolution:
        """
        Optimise the given function with given constraints.
        Main function of the Simplex class.

        Returns
        -------
        SimplexSolution
            solution of the optimization problem (vector of decision variables and optimal value)
        """
        if not self.to_maximize:
            self.C = -self.C

        prev_z = None
        while True:
            # X_B row b...
            # C_B row c

            # Step 1
            self._compute_basic_solution()
            # Step 2
            entering_j, min_delta, cnt = self._estimate_delta_row()
            optimal, solution = self._check_solution_for_optimality(cnt)

            if print_iterations:
                self.show_table()

            if prev_z is not None and round(self.z, self.eps) == round(prev_z, self.eps):
                raise AlternatingOptima(solution)

            prev_z = self.z
            if optimal:
                return solution

            # Step 3
            leaving_i, min_ratio, P_j = self._estimate_ratio_col(entering_j)

            # Step 4
            self._update_basis(entering_j, leaving_i, P_j)
            self.iteration += 1

    def show_table(self):
        print(f'Iteration #{self.iteration}')
        data = [[round(list(self.C_B)[row], self.eps), f'x_{self.basic[row] + 1}', round(list(self.X_B)[row], self.eps),
                 *[round(el, self.eps) for el in list(np.linalg.inv(self.B))[row]]] for row in
                range(self.m)] + [[' ', 'delta', round(self.delta, self.eps)] + self.z_row[len(self.function):]]

        view = tt.to_string(
            data,
            header=['c', 'basic', 'b'] + [f'A{i + 1}' for i in range(self.m)],
            style=tt.styles.rounded_thick
        )
        print(view)

    def _compute_basic_solution(self) -> None:
        """
        Compute basic solution at current iteration.
        Helper function for the optimise method.
        See Also:
            optimise
        """
        self.X_B = np.linalg.inv(self.B) @ self.b
        self.z = self.C_B @ self.X_B
        self.C_B_mul_B_inv = self.C_B @ np.linalg.inv(self.B)

    def _estimate_delta_row(self) -> tuple:
        """
        Estimate delta row at current iteration.
        Finds the entering variable.
        Helper function for the optimise method.
        See Also:
            optimise

        Returns
        -------
        entering_j: int
            index of the entering variable in the table

        min_delta: float
            minimum delta value in the row

        count: int
            number of positive deltas in the row
        """
        entering_var = 0
        min_delta = float("inf")
        positive_delta_count = 0
        self.z_row.clear()
        for column in range(self.n):
            P_j = self.A[:, [column]]
            z_j = self.C_B_mul_B_inv @ P_j
            delta = z_j.item() - self.C[column]
            self.z_row.append(round(delta, self.eps))
            if column in self.basic:
                continue

            if delta >= self.epsilon:
                positive_delta_count += 1
            elif delta < min_delta - self.epsilon:
                min_delta = delta
                entering_var = column

        return entering_var, min_delta, positive_delta_count

    def _estimate_ratio_col(self, entering_j):
        """
        Estimate ratio column at current iteration.
        Finds the leaving variable.
        Helper function for the optimise method.
        See Also:
            optimise
        Parameters
        ----------
        entering_j: int
            index of the entering variable in the table

        Returns
        -------
        leaving_i: int
            index of the leaving variable in the table

        min_ratio: float
            minimum ratio value in the column

        P_j: np.array
            column of the matrix A corresponding to the entering variable
        """
        P_j = self.A[:, [entering_j]]
        B_inv_mul_P_j = np.linalg.inv(self.B) @ P_j
        if np.all(B_inv_mul_P_j <= self.epsilon):
            raise InfeasibleSolution
        i = 0
        leaving_var = 0
        min_ratio = float("inf")
        for j in self.basic:
            if B_inv_mul_P_j[i] <= self.epsilon:
                i += 1
                continue
            ratio = self.X_B[i] / B_inv_mul_P_j[i].item()
            if ratio < min_ratio - self.epsilon:
                min_ratio = ratio
                leaving_var = j
            i += 1
        return leaving_var, min_ratio, P_j

    def _check_solution_for_optimality(self, count) -> tuple[bool, object]:
        """
        Check if the solution is optimal.
        Helper function for the optimise method.
        See Also:
            optimise
        Parameters
        ----------
        count: int
            number of positive deltas in the row

        Returns
        -------
        bool
            True if the solution is optimal, False otherwise

        SimplexSolution
            solution if the solution is optimal, None otherwise
        """
        x_decision = np.zeros(self.n - self.m)
        for i, j in enumerate(self.basic):
            if j < self.n - self.m:
                x_decision[j] = round(self.X_B[i], self.eps)
        self.delta = SimplexSolution(x_decision, round(self.z, self.eps)).opt
        if count == self.n - self.m:
            return True, SimplexSolution(x_decision, round(self.z, self.eps))
        return False, SimplexSolution(x_decision, round(self.z, self.eps))

    def _update_basis(self, entering_j, leaving_i, P_j) -> None:
        """
        Update basis at current iteration.
        Helper function for the optimise method.
        See Also:
            optimise
        Parameters
        ----------
        entering_j
        leaving_i
        P_j
        """
        for i in range(self.m):
            if self.basic[i] == leaving_i:
                self.B[:, [i]] = P_j
                self.basic[i] = entering_j
                self.C_B[i] = self.C[entering_j]
                break

    def _check_for_alternating_optima(self, delta_row) -> bool:
        """
        Check for alternating optima based on delta row.
        Helper function for the optimise method.
        See Also:
            optimise
        Parameters
        ----------
        delta_row: list
            List of delta values for each variable

        Returns
        -------
        bool
            True if alternating optima are detected, False otherwise
        """
        # Count the number of positive delta values
        num_positive_deltas = sum(1 for delta in delta_row if delta >= self.epsilon)

        # If there is more than one positive delta, it indicates alternating optima
        if num_positive_deltas > 1:
            return True
        return False


if __name__ == "__main__":
    from input_parser import parse_file

    simplex = Simplex(*parse_file("inputs/input1.txt"))
    np.set_printoptions(precision=simplex.eps)
    print(simplex)
    try:
        solution = simplex.optimise(print_iterations=True)
        print(solution)
    except InfeasibleSolution:
        print("SOLUTION:\nThe method is not applicable!")
    except AlternatingOptima as e:
        print(e.solution)
        print("Alternating optima detected")
