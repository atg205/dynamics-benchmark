from benchmarker.core import results_loader
from benchmarker.core import plotter
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Plot dynamics for a quantum system with specified parameters.'
    )
    parser.add_argument(
        '--system', 
        type=int,
        required=True,
        help='System ID to plot (1-8)'
    )
    parser.add_argument(
        '--timepoints',
        type=int,
        required=True,
        help='Number of time points in the simulation'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create plotter and generate plot
    plotter_instance = plotter.BenchmarkPlotter()
    plotter_instance.plot_dynamics(
        system=args.system,
        timepoints=args.timepoints
    )

if __name__ == "__main__":
    main()










