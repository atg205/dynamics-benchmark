from typing import Dict, Any
import pandas as pd 
import dimod

class BenchmarkResult:
    def __init__(self, result: dimod.SampleSet, system: int, ta: int, computation_time: int):
        self.result = result
        self.system = system
        self.ta = ta
        self.computation_time = computation_time