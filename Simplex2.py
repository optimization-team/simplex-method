from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import termtables as tt


@dataclass
class SimplexSolution:
    """Stores a solution to simplex method.

    Args:
        decision_variables (npt.NDArray[np.double]): Vector of decision variables
        value (float): Solution to maximization problem
    """

    x: npt.NDArray[np.double]
    opt: float


class InfeasibleSolution(Exception):
    """Raised when no feasible solution is found."""

    def __init__(self, message="No feasible solution is found"):
        self.message = message
        super().__init__(self.message)


class Simplex:
    def __init__(
            self,
            C: npt.NDArray,
            A: npt.NDArray,
            b: npt.NDArray,
            eps: float = 1e-4,
    ):
        """Initialization of simplex method solver.

        Args:
            A (npt.NDArray): (m x n) matrix representing the constraint coefficients
            b (npt.NDArray): m-vector representing the constraint right-hand side
            C (npt.NDArray): n-vector representing the objective-function coefficients
            eps (float, optional): Approximation accuracy. Defaults to 1e-4.
        """

        assert A.ndim == 2, "A is not a matrix"
        assert b.ndim == 1, "b is not a vector"
        assert C.ndim == 1, "C is not a vector"
        assert (
                A.shape[0] == b.size
        ), "Length of vector b does not correspond to # of rows of matrix A"
        assert (
                A.shape[1] == C.size
        ), "Length of vector C does not correspond to # of cols of matrix A"

        self.A = np.hstack(
            (A, np.identity(A.shape[0], dtype=np.double))
        )  # Adding slack variables
        self.C = np.hstack(
            (C, np.zeros(A.shape[0], dtype=np.double))
        )  # Adding slack variables
        self.b = b
        self.eps = eps
        self.m, self.n = self.A.shape

        # variables
        self.B = np.identity(self.m, dtype=np.double)
        self.C_B = np.zeros(self.m, dtype=np.double)
        self.basic = list(range(self.n - self.m, self.n))
        self.X_B = np.zeros(self.m, dtype=np.double)
        self.z = 0.0
        self.C_B_times_B_inv = np.zeros(self.m, dtype=np.double)

    def print_debug(self):
        print("-" * 10)
        print("BASIC:", self.basic)
        print("B:", self.B)
        print("X_B:", self.X_B)
        print("Z:", self.z)
        print("C_B:", self.C_B)
        print("C_B*B_inv", self.C_B_times_B_inv)

    def print_table(self, entering_j, leaving_i, P_j):
        pass

    def _compute_basic_solution(self):
        self.X_B = np.linalg.inv(self.B) @ self.b
        self.z = self.C_B @ self.X_B
        self.C_B_times_B_inv = self.C_B @ np.linalg.inv(self.B)

    def _estimate_delta_row(self):
        entering_j = 0
        min_delta = float("inf")
        count = 0
        for j in range(self.n):
            if j in self.basic:
                continue
            P_j = self.A[:, [j]]
            z_j = self.C_B_times_B_inv @ P_j
            delta = z_j.item() - self.C[j]
            if delta >= self.eps:  # maximization
                count += 1
            else:
                if delta < min_delta - self.eps:
                    min_delta = delta
                    entering_j = j
        return entering_j, min_delta, count

    def _estimate_ratio_col(self, entering_j):
        P_j = self.A[:, [entering_j]]
        B_inv_times_P_j = np.linalg.inv(self.B) @ P_j
        if np.all(B_inv_times_P_j <= self.eps):
            raise InfeasibleSolution
        i = 0
        leaving_i = 0
        min_ratio = float("inf")
        for j in self.basic:
            if B_inv_times_P_j[i] <= self.eps:
                i += 1
                continue
            ratio = self.X_B[i] / B_inv_times_P_j[i].item()
            if ratio < min_ratio - self.eps:
                min_ratio = ratio
                leaving_i = j
            i += 1
        return leaving_i, min_ratio, P_j

    def _check_solution_for_optimality(self, cnt):
        if cnt == self.n - self.m:
            round_decimals = round(-np.log10(self.eps))
            X_decision = np.zeros(self.n - self.m)
            for i, j in enumerate(self.basic):
                if j < self.n - self.m:
                    X_decision[j] = round(self.X_B[i], 3)
            return True, SimplexSolution(X_decision, round(self.z.item(), 3))
        else:
            return False, None

    def _update_basis(self, entering_j, leaving_i, P_j):
        for i in range(self.m):
            if self.basic[i] == leaving_i:
                self.B[:, [i]] = P_j
                self.basic[i] = entering_j
                self.C_B[i] = self.C[entering_j]
                break

    def optimise(self, debug: bool = False) -> SimplexSolution:
        while True:
            # Step 1
            self._compute_basic_solution()
            if debug:
                self.print_debug()

            # Step 2
            entering_j, min_delta, cnt = self._estimate_delta_row()
            optimal, solution = self._check_solution_for_optimality(cnt)
            if optimal:
                return solution

            # Step 3
            leaving_i, min_ratio, P_j = self._estimate_ratio_col(entering_j)

            # Step 4
            self._update_basis(entering_j, leaving_i, P_j)


def main():
    from input_parser import parse_file, parse_test
    from numpy import array, matrix

    function, A, b, approximation = parse_file('inputs/input4.txt')

    simplex = Simplex(array(function.coefficients), A, array(b), 1e-4)

    solution = simplex.optimise()
    print(solution)


if __name__ == "__main__":
    main()
