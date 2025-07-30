from typing import Dict, Any
import pandas as pd 

class BenchmarkResult:
    def __init__(self, result:pd.DataFrame,system:int, ta:int, computation_time:int):
        self.result =result
        self.system = system
        self.ta = ta
        self.computation_time = computation_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            'system': self.system,
            'ta': self.ta,
            'computation_time': self.computation_time,
            'result': self.result
        }
