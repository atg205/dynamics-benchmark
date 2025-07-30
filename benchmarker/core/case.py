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
    system: int
    solver: str
    precision: int
    timepoints: int

    def __init__(self, system: int, sampler: str, timepoints: int):
        self.system = system
        self.sampler = sampler
        self.timepoints = timepoints

    @abstractmethod
    def run(self) -> BenchmarkResult:
        pass

    def run_and_save(self) -> BenchmarkResult:
        result = self.run()
        save_benchmark_result(
            self.system,
            self.sampler,
            self.precision,
            self.timepoints,
            result
        )
        return result

class QuantumTestCase(TestCase):
    """Base class for quantum test cases."""
    name = "Quantum Benchmark"

    def __init__(self, system: int, sampler: str, timepoints: int,ta:int):
        super().__init__(system, sampler, timepoints)
        self.ta =ta

    def create_instance(self):
        """Create a benchmark instance for the current system."""
        return BenchmarkInstance(
            instance_id=int(self.system),
            number_time_points=self.timepoints,
        )

    def sample_qubo(self, qubo, num_samples: int = 1000, annealing_time: int = 200):
        """Sample from QUBO using quantum hardware."""
        if self.sampler is 'Neal':
            # If no sampler specified, use SimulatedAnnealingSampler
            sampler = neal.SimulatedAnnealingSampler()
            return sampler.sample(qubo, num_reads=num_samples)

        # Configure D-Wave sampler based on solver ID
        elif self.sampler == "5.4":
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system5.4", region="eu-central-1"))
        elif self.sampler == "1.4":  # zephyr
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage2_system1.4"))
        elif self.sampler == "6.4":
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system6.4"))
        else:
            raise ValueError(f"Invalid solver id: {self.solver}")

        return dw_sampler.sample(qubo, num_reads=num_samples, annealing_time=annealing_time, return_embedding=True)

    def compute_metrics(self, sampleset, instance: BenchmarkInstance = None) -> Dict[str, Any]:
        """Compute benchmark metrics from a sampleset."""
        pass

    def run(self) -> BenchmarkResult:
        """Common run logic for all quantum test cases."""
        instance = self.create_instance()
        sampleset = self.sample_qubo(instance.qubo)
        
        df = sampleset.to_pandas_dataframe() 
        computation_time = sampleset.info['timing']['qpu_access_time'] * 1e-3

        return BenchmarkResult(
            result = df,
            system = self.system,
            ta=self.ta,
            computation_time=computation_time
        ) 

class GroupTestCase(ABC):
    def __init__(self, system_list: list, num_timepoints: int, sampler: str, ta: int = 0):
        self.system_list = system_list
        self.sampler = sampler
        self.num_timepoints = num_timepoints
        self.ta = ta

    def run_all_systems(self):
        """Run benchmarks for all native systems."""
        results = []
        for system_id in self.system_list:
            self.system = system_id
            TC = QuantumTestCase(self.system, self.sampler, self.num_timepoints, self.ta)
            results.append(TC.run_and_save())
        return results



class PegasusNativeSystemsCase(GroupTestCase):
    name = "Pegasus Native Benchmark"
    def __init__(self,num_timepoints: int, sampler: str, ta:int=0):
        native_systems = [1, 2, 4, 5, 6, 7]
        super().__init__(native_systems,num_timepoints,sampler,ta)



class NonNativeTestCase(QuantumTestCase):
    name = "Non-Native Benchmark"
    non_native_systems = [3, 8]


    def get_system_type(self) -> str:
        return "non-native"

    def run_all_systems(self):
        """Run benchmarks for all non-native systems."""
        results = []
        for system_id in self.non_native_systems:
            self.system = system_id
            results.append(self.run_and_save())
        return results
