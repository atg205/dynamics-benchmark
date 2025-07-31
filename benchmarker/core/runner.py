from typing import List, Dict, Any
from .case import TestCase, PegasusNativeSystemsCase
from .results import BenchmarkResult

class BenchmarkRunner:
    """
    Runs a suite of TestCase instances and collects their results.
    """
    def __init__(self, test_cases: List[TestCase]):
        self.test_cases = test_cases
        self.results: List[BenchmarkResult] = []

    def run(self) -> List[BenchmarkResult]:
        self.results = []
        for case in self.test_cases:
            result = case.run()
            self.results.append(result)
        return self.results 

    def summary(self) -> Dict[str, Any]:
        return {
            'num_cases': len(self.test_cases),
            'results': [r.to_dict() for r in self.results]
        }


