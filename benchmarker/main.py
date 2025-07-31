from benchmarker.core.runner import BenchmarkRunner
from benchmarker.core.case import QuantumTestCase

# Create test cases
test_cases = [
    QuantumTestCase(
        system=1,
        sampler='neal',
        timepoints=3,
        ta=200
    )
]

# Run benchmarks
runner = BenchmarkRunner(test_cases=test_cases)
results = runner.run()

print("Done")