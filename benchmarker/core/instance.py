import os
import json
import pickle
from typing import Any
from dwdynamics import ComplexDynamicsProblem, Objective


class BenchmarkInstance:
    def __init__(self, instance_id: int, number_time_points: int, objective: Any = Objective.hessian, basepath: str = "", data_dir: str = "data/instances"):
        self.instance_id = instance_id
        self.objective = objective
        self.objective_path = 'norm' if objective == Objective.norm else 'hessian'
        self.number_time_points = number_time_points
        # Load data and create problem immediately
        file_name = os.path.join(basepath, data_dir, f"{self.instance_id}.pckl")
        print(os.getcwd())
        with open(file_name, 'rb') as f:
            instance_dict = pickle.load(f)
        self.H = instance_dict['H']
        self.psi0 = instance_dict['psi0']
        self.precision = instance_dict['precision']
        self.problem = ComplexDynamicsProblem(
            hamiltonian=self.H,
            initial_state=self.psi0,
            times=tuple(range(number_time_points)),
            num_bits_per_var=precision
        )
        self.qubo = self.problem.qubo(objective=self.objective)

    def save_qubo(self, basepath: str = "", save_dir: str = "data/instances"):
        if self.qubo is None:
            raise ValueError("QUBO not created yet.")
        path = os.path.join(basepath, save_dir, self.objective_path, str(self.instance_id))
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, f"precision_{self.precision}_timepoints_{self.number_time_points}.json")
        with open(file_name, 'w') as f:
            json.dump(self.qubo.to_serializable(), f)
