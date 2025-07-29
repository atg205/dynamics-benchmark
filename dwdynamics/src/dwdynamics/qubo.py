
"""Functions constructing QUBO used for matrix inversion."""
from enum import Enum
from itertools import product
import numpy as np
import numpy.typing as npt
from dimod import BQM, BINARY, BinaryQuadraticModel

from .operators import RealOperator, Vector


class Objective(str, Enum):
    norm = "norm"
    hessian = "hessian"


def real_linear_equation_qubo(
    coeff_matrix: RealOperator,
    rhs: Vector[np.float64],
    num_bits_per_var: int,
    objective: Objective = Objective.hessian,
    exp_offset: int = 0
):
    if objective == Objective.hessian:
        return real_symmetric_linear_equation_qubo(coeff_matrix, rhs, num_bits_per_var, exp_offset)
    else:
        return real_linear_equation_qubo_norm(coeff_matrix, rhs, num_bits_per_var, exp_offset)


def real_linear_equation_qubo_norm(
    coeff_matrix: npt.ArrayLike,
    rhs: npt.ArrayLike,
    num_bits_per_var: int,
    exp_offset: int = 0
) -> BQM:
    """Construct QUBO for solving linear equation Mx = Y, using norm-based objective-function.

    Args:
        coeff_matrix: System matrix M.
        rhs: right hand side Y of the equation to be solved.
        num_bits_per_var: Number of variables used for encoding each fixed point
            number.
        exp_offset: Exponent D defining the range of the variables. As per original
            construction, all coefficient of the unknown x will be encoded as
            fixed point numbers in the interval [-2 ** D, 2 ** D + 1].
            It is the caller responsibility to estimate required exp_offset,
            this function performs no heuristic whatsoever to determine the
            optimal value of this parameter. Default is 0, which is suitable
            for coefficients of quantum systems.

    Returns:
        A BQM object encapsulating QUBO for solving the linear equation Mx = Y,
        where x is a real vector of the same length as Y.
        The constructed QUBO contains len(rhs) * num_bits_per_var binary variables.
        Each consecutive num_bits_per_var binary variables correspond to a single
        logical fixed point number.
    """
    # Shorten variable names so that we don't go insane writing expressions.
    M = np.asarray(coeff_matrix).squeeze()
    Y = np.asarray(rhs).squeeze()
    R = num_bits_per_var
    D = exp_offset
    N = len(Y)


    if len(M.shape) != 2:
        raise ValueError(f"Only two dimensional coefficient matrices are supported. Shape passed: {M.shape}.")

    if coeff_matrix.shape != (len(Y), len(Y)):
        raise ValueError(
            "Coefficient matrix has to be square with dimension matchin lenght of the rhs. "
            f"Got coeff matrix with shape {M} and rhs of length {len(Y)}."
        )

    N = len(Y)

    # We have N (real) logical variables each with `num_bits_per_var` binary digits.
    # Thus, we have N * num_bits_per_var binary variables, indexed with pairs (i, digit).
    # The function q below maps this 2D index into a linear one. It is needed because
    # some solvers may require variables to be labelled as consecutive integers.
    def q(i: int, digit: int):
        return i * num_bits_per_var + digit
    


    # We first compute linear and quadratic coefficients (without constant)
    linear = dict[int, float]()
    quadratic = dict[(int, int), float]()

    # QUBO terms as derived in eq. (3.17) - (3.18)
    for i, r in product(range(N), range(R)):
        # Linear term
        linear[q(i, r)] = 4 * 2 ** (-r + D) * sum(
            M[k][i] * (2 ** (-r + D) * M[k][i] - Y[k] - 2 ** D * M[k].sum()) for k in range(N)
        )

        # Qudratic term
        for j, s in product(range(N), range(R)):
            if (j, s) != (i, r):
                quadratic_term = 4 * 2 ** (-r-s + D * 2) * sum(M[k][i] * M[k][j] for k in range(N))
                if abs(quadratic_term) > 1e-10 :
                    quadratic[(q(i, r), q(j, s))] = quadratic_term 

    # Next calculate the offset. This is so that we can easily compute the residuals of
    # the solution.
    offset = 2 ** (D * 2) * (M.T.dot(M).sum())
    offset += 2 ** (D + 1) * sum(M[j, i] * Y[j] for i, j in product(range(N), range(N)))
    offset += Y.dot(Y)
    return BQM(linear, quadratic, offset, BINARY)


def real_symmetric_linear_equation_qubo(
    coeff_matrix: npt.ArrayLike,
    rhs: npt.ArrayLike,
    num_bits_per_var: int,
    exp_offset: int = 0
) -> BQM:
    """Construct QUBO for solving linear equation Mx = Y, using hessian-based objective-function.

    Notes:
        This method requires coeff_matrix to be symmetric.

    Args:
        coeff_matrix: System matrix M.
        rhs: right hand side Y of the equation to be solved.
        num_bits_per_var: Number of variables used for encoding each fixed point
            number.
        exp_offset: Exponent D defining the range of the variables. As per original
            construction, all coefficient of the unknown x will be encoded as
            fixed point numbers in the interval [-2 ** D, 2 ** D + 1].
            It is the caller responsibility to estimate required exp_offset,
            this function performs no heuristic whatsoever to determine the
            optimal value of this parameter. Default is 0, which is suitable
            for coefficients of quantum systems.

    Returns:
        A BQM object encapsulating QUBO for solving the linear equation Mx = Y,
        where x is a real vector of the same length as Y.
        The constructed QUBO contains len(rhs) * num_bits_per_var binary variables.
        Each consecutive num_bits_per_var binary variables correspond to a single
        logical fixed point number.
    """
    # Shorten variable names so that we don't go insane writing expressions.
    M = np.asarray(coeff_matrix).squeeze()
    M = np.where(np.abs(M) < 1e-10, 0, M)
    Y = np.asarray(rhs).squeeze()
    R = num_bits_per_var
    D = exp_offset
    N = len(Y)
    linear = {}
    quadratic = {}

    if len(M.shape) != 2:
        raise ValueError(f"Only two dimensional coefficient matrices are supported. Shape passed: {M.shape}.")

    if coeff_matrix.shape != (len(Y), len(Y)):
        raise ValueError(
            "Coefficient matrix has to be square with dimension matchin lenght of the rhs. "
            f"Got coeff matrix with shape {M.shape} and rhs of length {len(Y)}."
        )
    # We have N (real) logical variables each with `num_bits_per_var` binary digits.
    # Thus, we have N * num_bits_per_var binary variables, indexed with pairs (i, digit).
    # The function q below maps this 2D index into a linear one. It is needed because
    # some solvers may require variables to be labelled as consecutive integers.
    def q(i: int, digit: int):
        return i * num_bits_per_var + digit

    # QUBO terms as derived in eq. (12) in our manuscript
    for i, r in product(range(N), range(R)):
        # Linear term
        linear[q(i, r)] = 2 ** (1 - r + D) * (
            (2 ** (-r + D) * M[i, i])
            - 2 ** D * (M[i].sum())
            - Y[i])

        # Qudratic term
        for j, s in product(range(N), range(R)):
            if (j, s) != (i, r):
                quadratic_term = M[i, j] * 2 ** (1 - r - s + 2 * D)
                if abs(quadratic_term) > 1e-10:
                    quadratic[(q(i, r), q(j, s))] = quadratic_term

    offset = 2 ** D * (2 ** (D - 1) * (M.sum()) + (Y.sum())) + 0.5
    return BQM(linear, quadratic, offset, BINARY)
