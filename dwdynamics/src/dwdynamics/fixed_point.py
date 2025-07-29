"""Utilities for working with fixed-point encoded numbers."""
from collections.abc import Mapping, Sequence
from typing import Callable

import numpy as np


def linear_index_map(stride_size: int) -> Callable[[int, int], int]:
    """Given a stride size, return a mapping of 2D indices into linear indices."""
    def _map(i: int, j: int) -> int:
        return i * stride_size + j
    return _map


def decode_number(bits: Sequence[int], exp_offset: int = 0) -> float:
    """Given bits of a fixed-point encoded number, decode it to float.

    Args:
        bits: Consecutive bits to decode into number.
        exp_offset: An exponent D determining range of the number, see the paper.

    Returns:
        Decoded number, as float.
    """
    number_decoded = 2 ** exp_offset * (2 * sum(2 ** (-alpha) * bit 
                                      for alpha, bit in enumerate(bits)) - 1)
    return number_decoded


def decode_sample(
    sample: Mapping[int, int],
    n_bits_per_var: int,
    exp_offset: int = 0
) -> Sequence[float]:
    """Decode sample (e.g. from D-Wave) encoding fixed-point vector into vector of real numbers.

    Args:
        sample: Either a sequence or a dictionary mapping consecutive integers (starting from zero),
            to values of binary variables.
        n_bits_per_var: Number of bits corresponding to a single real variable.
        exp_offset: Offset D determining range of real variables.

    Returns:
        A vector of floating point numbers.
    """
    N = len(sample) // n_bits_per_var
    q = linear_index_map(n_bits_per_var)
    result = np.zeros(N)
    for i in range(N):
        subsample = [sample[q(i,r)] for r in range(n_bits_per_var)]
        result[i] = decode_number(subsample, exp_offset)
    return result
