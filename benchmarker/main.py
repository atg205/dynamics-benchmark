from core.runner import BenchmarkRunner
from core.case import QuantumTestCase
def main():
    native_systems = [1, 2, 4, 5, 6, 7]
    test_cases = [QuantumTestCase(system,sampler='neal',timepoints=3,ta=200) for system in native_systems]
    runner = BenchmarkRunner(test_cases=test_cases)
    print(runner.run())


if __name__ == "__main__":
    main()