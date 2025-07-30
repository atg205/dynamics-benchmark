from typing import Dict, Any
import pandas as pd 

class BenchmarkResult:
    def __init__(self, result: pd.DataFrame, system: int, ta: int, computation_time: int):
        self.result = result
        self.system = system
        self.ta = ta
        self.computation_time = computation_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'system': self.system,
            'ta': self.ta,
            'computation_time': self.computation_time,
            'result': self.result.to_dict(orient='records')
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        return cls(
            result=pd.DataFrame(data['result']),
            system=data['system'],
            ta=data['ta'],
            computation_time=data['computation_time']
        )
    
    def get_expectation_values(self):
        """Return time and expectation value arrays"""
        return self.result['time'].values, self.result['expectation'].values