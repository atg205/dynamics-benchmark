from benchmarker.core.runner import BenchmarkRunner
from benchmarker.core.case import QuantumTestCase
from typing import Sequence

# Create test cases
test_cases= [
    QuantumTestCase(
        system=1,
        sampler='6.4',
        timepoints=3,
        ta=200,
        num_reps=100
    )
]

# Run benchmarks
runner = BenchmarkRunner(test_cases=test_cases)
results = runner.run_and_save()

print("Done")