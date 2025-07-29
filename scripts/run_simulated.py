from benchmarker.core.case import PegasusNativeTestCase

def main():
    # Create a test case with no sampler to use simulated annealing
    test_case = PegasusNativeTestCase(
        system="1",  # Using system 1 as an example
        solver="5.4",  # Solver ID is still needed for instance creation
        precision=4,  # Example precision
        timepoints=10,  # Example number of timepoints
        sampler=None  # This will trigger use of SimulatedAnnealingSampler
    )
    
    # Run the benchmark and print results
    result = test_case.run()
    print(f"Benchmark Results for {result.name}:")
    print(f"Metrics: {result.metrics}")
    print(f"Extra Info: {result.extra}")

if __name__ == "__main__":
    main()
