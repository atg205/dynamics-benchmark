"""Functions for constructing clocks operators."""
from collections.abc import Sequence
from itertools import islice
from typing import Callable 
import numpy as np
import scipy as sp
import pprint
from .operators import Operator, ComplexOperator, RealOperator, Propagator, T
import pprint


def explicit_propagator(hamiltonian: ComplexOperator, t2: float, t1: float) -> ComplexOperator:
    """Given a hamiltonian and two times t2 > t1, return a propagator explicitly computed via matrix exponentiation."""
    return sp.linalg.expm(1.0j *(t1 - t2) * hamiltonian)


def _truncate_imaginary(func: Propagator[np.complex128]) -> Propagator[np.float64]:
    def _inner(array: ComplexOperator, t2: float, t1: float) -> RealOperator:
        return func(array, t2, t1).real

    return _inner


def build_complex_clock(
    hamiltonian: ComplexOperator,
    times: Sequence[float],
    propagator: Propagator[np.complex128] = explicit_propagator
) -> ComplexOperator:
    """Construct a complex clock operator for given hamiltonian and given list of time points.

    Args:
        hamiltonian: Complex matrix defining the hamiltonian.
        times: A list of time points. It is assumed time points are given in chronological order.
        propagator: A function mapping (hamiltonian, t2, t1) to propagator U(t2, t1). By default,
            the propagator is computed by direct exponentiation, but you can override it with
            any approximation here. See `explicit_propagator` for signature of a function that
            can be passed as `propagator`.

    Returns:
        A complex matrix representing clock operator.
    """
    N = hamiltonian.shape[0]
    clock = 2 * np.eye(len(times) * N).astype(np.complex128)
    clock[-N:, -N:] = np.eye(N)
    _build_clock_in_place(clock, hamiltonian, times, propagator)
    return clock


def build_real_clock(
    hamiltonian: ComplexOperator,
    times: Sequence[float],
    propagator: Callable[[ComplexOperator, float, float], ComplexOperator] = explicit_propagator
) -> RealOperator:
    """Construct a real clock operator.

    This function differs from `build_complex_clock` in that it assumes that all propagators
    are real, and therefore discards the imaginary part of anything computed by `propagator`.
    """
    print(f"{hamiltonian.shape=}")
    N = hamiltonian.shape[0]
    clock = 2 * np.eye(len(times) * N)
    clock[-N:, -N:] = np.eye(N)
    _build_clock_in_place(clock, hamiltonian, times, _truncate_imaginary(propagator))
    print(f"{clock.shape=}")
    return clock


def _build_clock_in_place(
    empty_clock: Operator[T],
    hamiltonian: ComplexOperator,
    times: Sequence[float],
    propagator: Callable[[ComplexOperator, float, float], Operator[T]]
) -> None:
    N = hamiltonian.shape[0]
    for i, (t2, t1) in enumerate(zip(islice(times, 1, None), times)):
        U_t = propagator(hamiltonian, t2, t1)
        empty_clock[N * (i+1): N * (i+2), N * i: N * (i+1)] = -U_t
        empty_clock[N * i: N * (i+1), N * (i+1): N * (i+2)] = -U_t.conj().T
