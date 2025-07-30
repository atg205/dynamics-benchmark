# Example usage in a script
from benchmarker.core import results_loader
from benchmarker.core import plotter

# Load results
loader = results_loader.ResultsLoader()
results = loader.load_all_results(system=1)

# Create plots
plotter = plotter.BenchmarkPlotter()

# Plot single result
result = results['velox'][0]  # first result for velox solver
plotter.plot_dynamics(result)

# Plot comparison
plotter.plot_comparison(
    [results['velox'][0], results['1.4'][0]], 
    labels=['Velox', 'DWave 1.4']
)