from typing import Dict, List, Optional
import os
import json
import pandas as pd
from .results import BenchmarkResult
import numpy as np
import dimod
from collections import defaultdict
import re
import math
from scipy.stats import linregress


class ResultsLoader:
    def __init__(self, base_path: str = "benchmarker/data/results/hessian"):
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
    

    def return_tts(self,p_success: float,t:float, p_target=0.99)->float:
        """
        Calculate the time-to-solution (TTS) for a given success probability.

        Args:
            p_success (float): Success probability per run.
            t (float): Time per run.
            p_target (float): Target cumulative success probability (default 0.99).

        Returns:
            float: Estimated time to reach target success probability.
        """
        if p_success == 0:
            return np.inf
        if p_success == 1:
            return t
        return (math.log(1-p_target) / math.log(1-p_success))*t


    def get_dwave_tts(self, system: int,topology="6.4",file_limit=np.inf)->pd.DataFrame:

        path = os.path.join(self.base_path, str(system),topology)
        df_dict = defaultdict(list)
        file_counter = 0
        for file in os.listdir(path):
            #if file_counter >= file_limit:
            #   break
            with open(os.path.join(path,file),'r') as f:
                s = dimod.SampleSet.from_serializable(json.load(f))
            
            # Append Metadata
            qpu_access_time = s.info['timing']['qpu_access_time'] * 1e-3
            annealing_time = s.info['timing']['qpu_anneal_time_per_sample']
            precision = int(re.findall(r'(?<=precision_)\d+',file)[0])
            timepoints = int(re.findall(r'(?<=timepoints_)\d+',file)[0])
            
            df_dict['runtime'].append(qpu_access_time)
            df_dict['ta'].append(annealing_time)
            df_dict['precision'].append(precision)
            df_dict['timepoints'].append(timepoints)
            df_dict['num_var'].append(len(s.variables))

            sampleset = s.to_pandas_dataframe()
            sampleset['energy'] = abs(round(sampleset['energy'],10))
            success = len(sampleset[sampleset.energy== 0]) > 0
            
            df_dict['success'].append(success)
        
        df = pd.DataFrame.from_dict(df_dict)
        #return df
        df =df.groupby(['ta','precision','timepoints','num_var']).agg(
            success_sum=('success','sum'),
            runtime=('runtime','mean'),
            shots=('success','count')
        ).reset_index()
        df['success_prob'] = df['success_sum'] / df['shots']
        df['tts99'] = df.apply(lambda row: self.return_tts(row['success_prob'],row.runtime),axis=1)
        df = df[['precision','timepoints','num_var','tts99']].groupby(['precision','timepoints','num_var']).min().reset_index()
        df['source'] = topology
        df['system'] = system
        return df
    

    def get_velox_results(self,system: int)->pd.DataFrame:
        """
        Load Velox results for a given system from CSV and parse relevant fields.

        Args:
            system (int): System identifier.

        Returns:
            pd.DataFrame: DataFrame containing parsed results.
        """
        path = os.path.join(self.base_path, f'{system}/velox/best_results_hessian_{system}_native.csv')
        df = pd.read_csv(path)
        df_dict= defaultdict(list)
        for row in df.itertuples():
            precision, timepoints = re.findall(r'\d+',str(row.instance))
            df_dict['precision'].append(int(precision))
            df_dict['timepoints'].append(int(timepoints))
            df_dict['num_steps'].append(int(row.num_steps))
            df_dict['runtime'].append(float(row.runtime)*1e3)
            df_dict['gap'].append(float(row.gap))
            df_dict['num_rep'].append(int(row.num_rep))
            df_dict['success_prob'].append(float(row.success_prob))
            df_dict['solution'].append(row.best_solution.replace("-1","0").replace(';',''))
            df_dict['num_var'].append(int(row.num_var))
        return pd.DataFrame(df_dict)

    def get_velox_tts(self,system:int)->pd.DataFrame:
        """
        Compute Velox success rates for a given system.

        Args:
            system (int): System identifier.

        Returns:
            pd.DataFrame: DataFrame containing aggregated success rates and runtimes.
        """
        df = self.get_velox_results(system=system)
        df['tts99'] = df.apply(lambda row: self.return_tts(row['success_prob'],row.runtime),axis=1)
        df = df[['precision','timepoints','num_var','tts99']].groupby(['precision','timepoints','num_var']).min().reset_index()
        df['system'] = system
        df['source'] = 'VELOX'
        return df
    

    def get_dwave_success_rates(self,system: int,topology="6.4",ta=200,grouped=True,file_limit=np.inf)->pd.DataFrame:
        """
        Load D-Wave success rates for a given system and topology.

        Args:
            system (int): System identifier.
            topology (str): Topology string.
            ta (int): Annealing time.
            grouped (bool): If True, group results by precision, timepoints, topology.
            file_limit (int): Maximum number of files to process.

        Returns:
            pd.DataFrame: DataFrame containing success rates and related info.
        """
        path = os.path.join(self.base_path, str(system),topology)
        dfs = []
        df_dict = defaultdict(list)
        for topology in [topology]:
            path = os.path.join(self.base_path, str(system),topology)

            file_counter = 0
            for file in os.listdir(path):
                if file_counter >= file_limit:
                    break
                with open(os.path.join(path,file),'r') as f:
                    s = dimod.SampleSet.from_serializable(json.load(f))
                
                if topology=='neal':
                    qpu_access_time = s.info['timing']['sampling_ns']
                    annealing_time = 0
                else:
                    qpu_access_time = s.info['timing']['qpu_access_time']
                    annealing_time = s.info['timing']['qpu_anneal_time_per_sample']
                    if not ta == annealing_time:
                        continue
                
                file_counter+=1
                
                df_dict['precision'].append(int(re.findall(r'(?<=precision_)\d+',file)[0]))
                df_dict['timepoints'].append(int(re.findall(r'(?<=timepoints_)\d+',file)[0]))
                df = s.to_pandas_dataframe()
                df['energy'] = abs(round(df['energy'],10))
                df = df[['energy','num_occurrences']].groupby(by=["energy"]).sum().reset_index()
                if len(df[df.energy== 0]) == 0:
                    success_rate = 0.0
                else:
                    success_rate = int(df[df.energy == 0]['num_occurrences'].iloc[0])
                success_rate /= df['num_occurrences'].sum()
                access_time = qpu_access_time / df['num_occurrences'].sum() * 1e-3
                df_dict['topology'].append(topology)
                df_dict['success_prob'].append(success_rate)
                df_dict['runtime'].append(access_time)
                df_dict['num_var'].append(len(s.variables))
                dfs.append(pd.DataFrame(df_dict))
        dfs_all = pd.concat(dfs)
        if grouped:
            dfs_all = dfs_all.groupby(['precision','timepoints','topology']).mean().reset_index()
        return dfs_all
    

    def get_dwave_sample_set(self,system: int, timepoints: int, topology="1.4") -> pd.DataFrame:
        path = os.path.join(self.base_path, str(system),topology)
        min_energy = np.inf
        return_sample =None
        for file in os.listdir(path):
            file_tp = int(re.findall(r'(?<=timepoints_)\d+',file)[0])
            if not file_tp == timepoints:
                continue

            with open(os.path.join(path,file),'r') as f:
                s = dimod.SampleSet.from_serializable(json.load(f))
            
            if np.round(s.first.energy,12) == 0:
                return s
            elif s.first.energy < min_energy:
                return_sample = s
                min_energy =s.first.energy
        if return_sample is None:
            raise ValueError('No samples for system/timepoints/topology combination found')
        else:
            return return_sample
