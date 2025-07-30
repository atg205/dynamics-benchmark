from typing import Dict, List, Optional
import os
import json
import pandas as pd
from .results import BenchmarkResult

class ResultsLoader:
    def __init__(self, base_path: str = "results"):
        self.base_path = base_path
    
    def load_result(self, system: int, solver: str, precision: int, timepoints: int) -> Optional[BenchmarkResult]:
        """Load a specific benchmark result"""
        path = os.path.join(self.base_path, str(system), str(solver), 
                          f"precision_{precision}_timepoints_{timepoints}.json")
        if not os.path.exists(path):
            return None
            
        with open(path, 'r') as f:
            data = json.load(f)
            return BenchmarkResult(
                result=pd.DataFrame(data['result']),
                system=data['system'],
                ta=data['ta'],
                computation_time=data['computation_time']
            )
    
    def load_all_results(self, system: int) -> Dict[str, List[BenchmarkResult]]:
        """Load all results for a given system"""
        results = {}
        system_path = os.path.join(self.base_path, str(system))
        
        if not os.path.exists(system_path):
            return results
            
        for solver in os.listdir(system_path):
            results[solver] = []
            solver_path = os.path.join(system_path, solver)
            
            for result_file in os.listdir(solver_path):
                if result_file.endswith('.json'):
                    with open(os.path.join(solver_path, result_file), 'r') as f:
                        data = json.load(f)
                        results[solver].append(BenchmarkResult(
                            result=pd.DataFrame(data['result']),
                            system=data['system'],
                            ta=data['ta'],
                            computation_time=data['computation_time']
                        ))
                        
        return results