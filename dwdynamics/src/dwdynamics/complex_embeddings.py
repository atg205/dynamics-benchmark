"""Tools for working with complex embeddings.

The standard embedding of complex numbers into R^2_2 is defined as
a + b*i -> [[re, im], [-im, re]]

Similarly, any complex matrix in C^N_N can be embedded into R^2N_2N
by embedding each coefficient and then flattening the array.
"""

from itertools import product
import numpy as np

from .operators import ComplexOperator, RealOperator


def embed_complex(val: ComplexOperator | complex) -> RealOperator:
    """Given a complex number or a complex matrix, represent it as a real matrix."""
    complex_arr = np.asarray(val)

    return _cplx_to_mat(complex_arr) if complex_arr.shape == () else _cplxmat_to_matmat(complex_arr)


def _cplx_to_mat(a: complex) -> RealOperator:
    """Converts a complex number into its matrix representation"""
    re = a.real
    im = a.imag
    return np.array([[re, im], [-im, re]])


def _cplxmat_to_matmat(cm: ComplexOperator) -> RealOperator:
    """Convert a complex NxN matrix into its real representation."""
    mm = np.empty((2 * cm.shape[0], 2 * cm.shape[1]), dtype=np.float64)
    for i, j in product(range(0, cm.shape[0]), range(0, cm.shape[1])):
        mm[2*i:2*(i+1), 2*j:2*(j+1)] = _cplx_to_mat(cm[i, j])
    return mm


