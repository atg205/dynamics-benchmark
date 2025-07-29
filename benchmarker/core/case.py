from abc import ABC, abstractmethod
from typing import Any, Dict
from dwave.system import DWaveSampler, EmbeddingComposite
import neal
import qutip as qp
import numpy as np
from .results import BenchmarkResult
from .save_utils import save_benchmark_result
from .instance import BenchmarkInstance

class TestCase(ABC):
    name: str
    system: str
    solver: str
    precision: int
    timepoints: int

    def __init__(self, system: str, solver: str, precision: int, timepoints: int):
        self.system = system
        self.solver = solver
        self.precision = precision
        self.timepoints = timepoints

    @abstractmethod
    def run(self) -> BenchmarkResult:
        pass

    def run_and_save(self) -> BenchmarkResult:
        result = self.run()
        save_benchmark_result(
            self.system,
            self.solver,
            self.precision,
            self.timepoints,
            result.to_dict()
        )
        return result

class QuantumTestCase(TestCase, ABC):
    """Abstract base class for quantum test cases."""
    name = "Quantum Benchmark"

    def __init__(self, system: str, solver: str, precision: int, timepoints: int, objective=None, sampler=None):
        super().__init__(system, solver, precision, timepoints)
        self.objective = objective
        self.sampler = sampler

    def create_instance(self):
        """Create a benchmark instance for the current system."""
        return BenchmarkInstance(
            instance_id=int(self.system),
            precision=self.precision,
            number_time_points=self.timepoints,
            objective=self.objective
        )

    @abstractmethod
    def sample_qubo(self, qubo, num_samples: int = 1000, annealing_time: int = 200):
        """Sample from QUBO using quantum hardware."""
        if self.sampler is None:
            # If no sampler specified, use SimulatedAnnealingSampler
            sampler = neal.SimulatedAnnealingSampler()
            return sampler.sample(qubo, num_reads=num_samples)

        # Configure D-Wave sampler based on solver ID
        if self.solver == "5.4":
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system5.4", region="eu-central-1"))
        elif self.solver == "1.4":  # zephyr
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage2_system1.4"))
        elif self.solver == "6.4":
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system6.4"))
        else:
            raise ValueError(f"Invalid solver id: {self.solver}")

        return dw_sampler.sample(qubo, num_reads=num_samples, annealing_time=annealing_time, return_embedding=True)

    def verify_sample(self, instance: BenchmarkInstance, sample: str) -> bool:
        """Verify a sample against the baseline quantum expectation values."""
        SZ = np.array([[1, 0], [0, -1]])
        exact_vec = instance.problem.interpret_sample(sample)
        exact_expect = [(state.conj() @ SZ @ state).real for state in exact_vec]
        times = list(range(instance.number_time_points))
        baseline = qp.mesolve(qp.Qobj(instance.H), qp.basis(2, 0), times, e_ops=[qp.sigmaz()]).expect[0]
        return np.allclose(baseline, exact_expect)

    @abstractmethod
    def compute_metrics(self, sampleset, instance: BenchmarkInstance = None) -> Dict[str, Any]:
        """Compute benchmark metrics from a sampleset."""
        pass

    def run(self) -> BenchmarkResult:
        """Common run logic for all quantum test cases."""
        instance = self.create_instance()
        sampleset = self.sample_qubo(instance.qubo)
        
        # Get the lowest energy sample
        sample = min(sampleset.samples(), key=lambda x: sampleset.energy(x))
        sample_str = "".join(str(bit) for bit in sample)
        
        # Compute all metrics including verification
        metrics = self.compute_metrics(sampleset, instance)
        metrics.update({
            "verified": self.verify_sample(instance, sample_str),
            "energy": float(sampleset.energies[0]),  # lowest energy
            "num_samples": len(sampleset),
        })
        
        return BenchmarkResult(
            name=self.name,
            metrics=metrics,
            extra={
                'system_type': self.get_system_type(),
                'best_sample': sample_str
            }
        )

    @abstractmethod
    def get_system_type(self) -> str:
        """Return the type of system (native/non-native)."""
        pass

class PegasusNativeTestCase(QuantumTestCase):
    name = "Pegasus Native Benchmark"
    native_systems = [1, 2, 4, 5, 6, 7]

    def sample_qubo(self, qubo, num_samples: int = 1000, annealing_time: int = 200):
        if self.sampler is None:
            return {"mock": "sampleset", "num_samples": num_samples}
        return self.sampler.sample(qubo, num_reads=num_samples, annealing_time=annealing_time)

    def compute_metrics(self, sampleset, instance: BenchmarkInstance = None) -> Dict[str, Any]:
        metrics = {
            "samples": len(sampleset) if hasattr(sampleset, "__len__") else sampleset.get("num_samples", 0),
        }
        
        # Add timing information if available
        if hasattr(sampleset, "info"):
            timing = sampleset.info.get("timing", {})
            metrics.update({
                "qpu_sampling_time": timing.get("qpu_sampling_time"),
                "qpu_programming_time": timing.get("qpu_programming_time"),
                "qpu_access_time": timing.get("qpu_access_time"),
                "total_post_processing_time": timing.get("total_post_processing_time")
            })
            
            # Add embedding info if available
            embedding_info = sampleset.info.get("embedding_context", {})
            metrics.update({
                "chain_break_fraction": embedding_info.get("chain_break_fraction"),
                "embedding_time": embedding_info.get("embedding_time")
            })
        
        return metrics

    def get_system_type(self) -> str:
        return "native"

    def run_all_systems(self):
        """Run benchmarks for all native systems."""
        results = []
        for system_id in self.native_systems:
            self.system = str(system_id)
            results.append(self.run_and_save())
        return results

class NonNativeTestCase(QuantumTestCase):
    name = "Non-Native Benchmark"
    non_native_systems = [3, 8]

    def sample_qubo(self, qubo):
        if self.sampler is None:
            return {"mock": "sampleset", "num_samples": 42}
        return self.sampler.sample(qubo)

    def compute_metrics(self, sampleset) -> Dict[str, Any]:
        return {
            "samples": sampleset.get("num_samples", 42),
            "embedding_time": 0.3,  # Mock for now
            "sampling_time": 0.4,  # Mock for now
            "minor_embedding_size": 100  # Mock for now
        }

    def get_system_type(self) -> str:
        return "non-native"

    def run_all_systems(self):
        """Run benchmarks for all non-native systems."""
        results = []
        for system_id in self.non_native_systems:
            self.system = str(system_id)
            results.append(self.run_and_save())
        return results
