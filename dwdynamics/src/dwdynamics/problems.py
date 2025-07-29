"""Definitions of ready-to-use problems.

The problem objects encapsulate all the data:
    - Hamiltonian
    - Initial state
    - Time points
    - Number of bits per logical (real) variable
    - Exponent offset.


All problems defined here have `to_qubo` method which converts them to a BQM object
suitable for being used with D-Wave (or any Dimod-based sampler really). The also
have `interpret_sample` method for converting a (binary) sample into a meaningful
solution to the given problem.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

from dimod import BQM
import numpy as np

from dwdynamics.complex_embeddings import embed_complex
from dwdynamics.qubo import real_linear_equation_qubo

from .clock import build_complex_clock, build_real_clock, explicit_propagator
from .fixed_point import decode_sample
from .operators import ComplexOperator, ComplexVector, Propagator, RealOperator, RealVector, Vector
from .qubo import Objective


@dataclass(frozen=True)
class RealLinearEquationProblem:
    """Solving linear equation of the form Ax = y."""
    coeff_matrix: RealOperator
    rhs: RealVector
    num_bits_per_var: int
    exp_offset: int = 0

    def qubo(self, objective: Objective = Objective.norm) -> BQM:
        """Convert this problem into QUBO."""
        return real_linear_equation_qubo(
            self.coeff_matrix,
            rhs=self.rhs,
            num_bits_per_var=self.num_bits_per_var,
            objective=objective,
            exp_offset=self.exp_offset
        )

    def interpret_sample(self, sample: Mapping[int, int]) -> Vector[np.float64]:
        """Given a binary sample, convert it to a vector of real numbers representing solution to this problem."""
        return np.asarray(decode_sample(sample, self.num_bits_per_var))


@dataclass(frozen=True)
class RealDynamicsProblem:
    """Solving dynamics of a real system with given hamiltonian."""
    hamiltonian: ComplexOperator
    initial_state: Vector[np.float64]
    times: Sequence[float]
    num_bits_per_var: int
    exp_offset: int = 0
    propagator: Propagator[np.complex128] = explicit_propagator

    def to_linear_eq_problem(self):
        """Convert this dynamics problem to a problem of solving real linear equation."""
        return self._linear_eq_problem

    @cached_property
    def _linear_eq_problem(self):
        N = len(self.initial_state)
        clock = build_real_clock(self.hamiltonian, self.times, self.propagator)
        rhs = np.hstack([self.initial_state.squeeze(), np.zeros(N * (len(self.times) - 1))])
        return RealLinearEquationProblem(
            coeff_matrix=clock,
            rhs=rhs,
            num_bits_per_var=self.num_bits_per_var,
            exp_offset=self.exp_offset
        )

    def qubo(self, objective: Objective = Objective.norm, propagator: Propagator[np.complex128] = explicit_propagator) -> BQM:
        """Create a QUBO corresponding to this problem, optionally overriding how the propagators are computed."""
        return self.to_linear_eq_problem().qubo(objective=objective)

    def interpret_sample(self, sample: Mapping[int, int]) -> Vector[np.float64]:
        """Given a sampler (e.g. from the annealer) interpret it as a solution to this problem.

        Solutions to this problem are floating point arrays of shape (len(self.times), N), where N
        is dimensionality of the system.
        """
        return self.to_linear_eq_problem().interpret_sample(sample).reshape(len(self.times), -1)


@dataclass(frozen=True)
class ComplexDynamicsProblem:
    """Solving dynamics of a complex system with given hamiltonian."""
    hamiltonian: ComplexOperator
    initial_state: ComplexVector
    times: Sequence[float]
    num_bits_per_var: int
    exp_offset: int = 0
    propagator: Propagator[np.complex128] = explicit_propagator
    clock: np.ndarray = None  # Optional user-supplied clock

    def get_clock(self):
        """Return the clock matrix used for this problem."""
        if self.clock is not None:
            return self.clock
        N = len(self.initial_state)
        return build_complex_clock(self.hamiltonian, self.times, self.propagator)

    def set_clock(self, clock):
        """Return a new instance with the given clock matrix set."""
        return self.__class__(
            hamiltonian=self.hamiltonian,
            initial_state=self.initial_state,
            times=self.times,
            num_bits_per_var=self.num_bits_per_var,
            exp_offset=self.exp_offset,
            propagator=self.propagator,
            clock=clock
        )

    def qubo(self, objective: Objective = Objective.norm, propagator: Propagator[np.complex128] = explicit_propagator) -> BQM:
        """Create a QUBO corresponding to this problem, optionally overriding how the propagators are computed."""
        return self.to_linear_eq_problem().qubo(objective=objective)

    def to_linear_eq_problem(self) -> RealLinearEquationProblem:
        """Convert this dynamics problem to a problem of solving real linear equation."""
        return self._linear_eq_problem


   
    @cached_property
    def _linear_eq_problem(self) -> RealLinearEquationProblem:
        N = len(self.initial_state)
        clock = self.get_clock()
        rhs = np.hstack([self.initial_state.squeeze(), np.zeros(N * (len(self.times)-1), dtype=np.complex128)])
        real_rhs = [comp for x in rhs for comp in (x.real, x.imag)]
        #clock = np.vectorize(lambda x: x.real)(clock)
        return RealLinearEquationProblem(
            coeff_matrix=embed_complex(clock),
            #coeff_matrix=clock,
            rhs=real_rhs,
            num_bits_per_var=self.num_bits_per_var,
            exp_offset=self.exp_offset
        )

    def interpret_sample(self, sample: Mapping[int, int]) -> ComplexVector:
        """Given a sampler (e.g. from the annealer) interpret it as a solution to this problem.

        Solutions to this problem are complex arrays of shape (len(self.times), N), where N
        is dimensionality of the system.
        """
        flat_sol = self.to_linear_eq_problem().interpret_sample(sample)
        cplx_sol = flat_sol[0::2] + 1j * flat_sol[1::2]
        return cplx_sol.reshape(len(self.times), -1)
