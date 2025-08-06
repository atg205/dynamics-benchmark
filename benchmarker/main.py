from benchmarker.core.runner import BenchmarkRunner
from benchmarker.core.case import QuantumTestCase
from benchmarker.core.plotter import BenchmarkPlotter
from typing import Sequence

# Create test cases

test_cases=[
    QuantumTestCase(
        system=system,
        sampler=sampler,
        timepoints=timepoints,
        ta=200,
        num_reps=1000
    ) 
    for system in [4,5,6,7] 
    for timepoints in range(2,6) 
    for sampler in ['1.4','6.4'] 
    for _ in range(5)
]



# Run benchmarks
runner = BenchmarkRunner(test_cases=test_cases)
results = runner.run_and_save()


BenchmarkPlotter().plot_tts(num_reps=1000, file_limit=20)
print("Done")