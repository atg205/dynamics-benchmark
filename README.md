# Quantum Dynamics Benchmarking Framework

**Authors:** Philipp Hanussek, Jakub Pawłowski, Zakaria Mzaouali, Bartłomiej Gardas

This repository accompanies the paper *Solving quantum-inspired dynamics on quantum and classical computers*.

A framework for benchmarking different approaches to solving quantum dynamics problems, with a focus on comparing quantum annealing and classical methods.



## Overview

This project provides tools for:
- Benchmarking quantum dynamics solvers (DWave, Velox, etc.)
- Collecting and analyzing performance metrics
- Visualizing results and dynamics
- Comparing different solution approaches

## Installation

### Requirements
- Python 3.11+
- Poetry (recommended) or pip

It is recommended to use a python virtual environment.

### Solver access
To perform calculation, solver API access is required. See [D-Wave help pages](https://support.dwavesys.com/hc/en-us/articles/360003682634-How-Do-I-Get-an-API-Token).
### Using Poetry (recommended)
```bash
# Clone the repository
git clone https://github.com/atg205/dynamics-benchmark.git

# Install poetry if you haven't already
pip install poetry

# Install dependencies
cd dynamics-benchmark/benchmarker
poetry install

# For development (editable install)
pip install -e .
```

## Project Structure
```
dynamics-benchmark/
├── benchmarker/           # Main package
│   ├── core/             # Core functionality
│   │   ├── case.py       # Test case definitions
│   │   ├── instance.py   # Problem instances
│   │   ├── results.py    # Results handling
│   │   ├── runner.py     # Benchmark execution
│   │   └── save_utils.py # Data persistence
│   └── main.py           # Entry point
├── data/                 # Benchmark data
│   ├── instances/        # Problem instances
│   ├── results/         # Benchmark results
│   └── xubo/            # XUBO formulations
└── plots/               # Generated visualizations
```

## Usage

### Basic Example
```python
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
```

### Plotting Results
```python
from benchmarker.core.plotter import BenchmarkPlotter

plotter = BenchmarkPlotter()
plotter.plot_dynamics(system=1, timepoints=3)
```

## Data Management

Results are stored in a hierarchical structure:
```
data/
├── instances/           # Raw problem instances
├── results/            # Benchmark results
│   └── <system_id>/
│       └── <solver>/
│           └── precision_<p>_timepoints_<t>.json
└── xubo/              # XUBO representations
```

## Related Projects

This project builds on [dwdynamics](dwdynamics/README.md), which provides the core functionality for quantum dynamics simulation. While dwdynamics focuses on the implementation of different solving methods, this framework adds benchmarking capabilities and result analysis.
