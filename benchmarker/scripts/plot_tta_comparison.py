from benchmarker.core import results_loader
from benchmarker.core import plotter

plotter = plotter.BenchmarkPlotter()
plotter.plot_tts(file_limit=5,num_reps=1000)
