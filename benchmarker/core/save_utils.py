from pathlib import Path
import json
from typing import Dict, Any
from .results import BenchmarkResult
from ..config import RESULTS_DIR, ensure_dir


def save_benchmark_result(system: int, solver: str, precision: int, timepoints: int, result: BenchmarkResult) -> None:
    """
    Save benchmark result to the configured results directory.
    
    Args:
        system: system or instance id/name
        solver: solver id (e.g., '6.4', '1.4', 'velox')
        precision: precision value (int)
        timepoints: number of timepoints (int)
        result: BenchmarkResult instance to save
    """
    # Construct path using pathlib
    result_path = RESULTS_DIR / str(system) / str(solver)
    ensure_dir(result_path)
    
    file_path = result_path / f"precision_{precision}_timepoints_{timepoints}.json"
    
    # Save result as JSON
    with file_path.open("w") as f:
        json.dump(result.to_dict(), f, indent=2)
