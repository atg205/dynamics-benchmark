import os
import json
from typing import Dict, Any
from .results import BenchmarkResult


def save_benchmark_result(system: int, solver: str, precision: int, timepoints: int, result: BenchmarkResult):
    """
    Save benchmark result to results/<system>/<solver>/precision_<p>_timepoints_<t>.json
    Args:
        system: system or instance id/name
        solver: solver id (e.g., '6.4', '1.4', 'velox')
        precision: precision value (int)
        timepoints: number of timepoints (int)
        result: dictionary to save (e.g., BenchmarkResult.to_dict())
    """
    dir_path = os.path.join("results", str(system), str(solver))
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"precision_{precision}_timepoints_{timepoints}.json")
    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)
