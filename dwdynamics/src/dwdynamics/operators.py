from typing import TypeVar, Callable, TypeAlias
import numpy as np

T = TypeVar("T", np.float64, np.complex128, covariant=True)

Operator = np.ndarray[tuple[int, int], np.dtype[T]]
RealOperator = Operator[np.float64]
ComplexOperator = Operator[np.complex128]

# "Propagator" is an operator, but here we abuse the name to mean "function that computes propagator",
# Just so that we don't call it `PropagatorFactory` or something.
Propagator = Callable[[ComplexOperator, float, float], Operator[T]]

Vector: TypeAlias = np.ndarray[tuple[int], np.dtype[T]]
RealVector: TypeAlias = Vector[np.float64]
ComplexVector: TypeAlias = Vector[np.complex128]

