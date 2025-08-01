"""Configuration file for path management in the benchmarker package."""
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent

# Define important directories
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = DATA_DIR / 'results'
INSTANCES_DIR = DATA_DIR / 'instances'
PLOTS_DIR = PROJECT_ROOT / 'plots'

# Common subdirectories
HESSIAN_RESULTS_DIR = RESULTS_DIR / 'hessian'
XUBO_DIR = DATA_DIR / 'xubo'

def get_results_dir(system_id: int, solver: str = None, topology: str = None) -> Path:
    """Get the results directory for a specific system and solver configuration."""
    path = HESSIAN_RESULTS_DIR / str(system_id)
    if solver:
        path = path / solver
    if topology:
        path = path / topology
    return path

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return its Path object."""
    path.mkdir(parents=True, exist_ok=True)
    return path
