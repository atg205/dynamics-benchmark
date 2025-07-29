from typing import Dict, Any

class BenchmarkResult:
    def __init__(self, name: str, metrics: Dict[str, Any], extra: Dict[str, Any] = None):
        self.name = name
        self.metrics = metrics
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'metrics': self.metrics,
            'extra': self.extra
        }
