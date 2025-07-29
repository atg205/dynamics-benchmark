import numpy as np
import os
import pickle
import qutip as qp
import math
import functools as ft
import re
import pandas as pd
from collections import defaultdict
import json
import dimod
import matplotlib as mpl


# Publication-quality (TikZ-like) matplotlib style setup
def set_pub_style(scale=1.0, fontsize=10, grid = True):
    fig_width_pt = 246.0  # width in pt (e.g., for single-column in a paper)
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    eps_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "figure.figsize": fig_size,
        "axes.titlesize": fontsize,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": 'black',
    }
    mpl.rcParams.update(eps_with_latex)

def random_matrix(dims, hermitian=True):
    """
    Generate a random complex matrix of given dimensions.
    If hermitian=True, returns a Hermitian matrix.

    Args:
        dims (tuple): Dimensions of the matrix.
        hermitian (bool): Whether to return a Hermitian matrix.

    Returns:
        np.ndarray: Random (Hermitian) matrix.
    """
    A = np.random.uniform(-1, 1, dims) + 1.j * np.random.uniform(-1, 1, dims)
    if hermitian:
        return A + A.conj().T
    return A


def get_last_index(files: list[str]) -> int:
    """
    Returns the highest index found in a list of filenames ending with .json.

    Args:
        files (list of str): List of filenames.

    Returns:
        int: Highest index found, or 0 if list is empty.
    """
    if not files:
        return 0
    return max([int(re.findall(r'\d+(?=\.json)', file)[0]) for file in files])


def get_instance(id, print_desc = True):
    """
    Retrieve an instance from the data/instances directory.

    Args:
        id (int or str): Instance identifier.
        print_desc (bool): If True, prints the instance description.

    Returns:
        dict: Loaded instance data.
    """
    path = f"/home/atg205/Documents/__Dokumente/Uni/UPMC/stage gl/DWaveDynamics2/data/instances"
    file_name = os.path.join(path, f"{id}.pckl")
    with open(file_name,'rb') as f:
        instance = pickle.load(f)
    if print_desc:
        print("-------")
        print(instance['about'])
        print("---------")
    return instance


def is_ptsymmetric(H):
    """
    Check if a Hamiltonian H is PT symmetric.

    Args:
        H (np.ndarray): Hamiltonian matrix.

    Returns:
        bool: True if PT symmetric, False otherwise.
    """
    dim = math.log(len(H),2)
    assert dim == int(dim)
    dim = int(dim)
    P = qp.sigmax().full()
    XX = ft.reduce(np.kron, [P]*dim)
    PT_H = XX @ H.conj() @ XX
    return np.allclose(H, PT_H)


def generate_pt_symmetric_real_eig(a):
    """
    Generate a 2D PT symmetric matrix with real eigenvalues ±1.

    Args:
        a (float): Parameter between 0 and 1 (exclusive).

    Returns:
        np.matrix: PT symmetric matrix.
    """
    assert a > 0 and a < 1, "a has to be between 0 and 1 exclusive"
    b = math.sqrt(1+a*a)
    H = np.matrix([[a*1.0j, b],[b,-a*1.0j]])
    assert is_ptsymmetric(H)
    return H

def generate_entangled_hamiltonian(num_qubits: int):
    """
    Create a Hamiltonian of the form H = |0...0><1...1| + |1...1><0...0|.

    Args:
        num_qubits (int): Number of qubits.

    Returns:
        np.ndarray: Entangled Hamiltonian matrix.
    """
    n = 2**num_qubits
    H = np.zeros((n, n))
    H[0, n-1] = 1
    H[n-1, 0] = 1
    return H

def generate_entangled_parity_hamiltonian(num_qubits: int):
    """
    Create a Hamiltonian of the form H = |1...0><0...1| + |0...1><1...0|.

    Args:
        num_qubits (int): Number of qubits.

    Returns:
        tuple: (Initial state vector, Hamiltonian matrix)
    """
    ket01 = qp.tensor([qp.basis(2, 0)]*(num_qubits-1)+[qp.basis(2, 1)])
    ket10 = qp.tensor([qp.basis(2, 1)]+[qp.basis(2, 0)]*(num_qubits-1))
    H = 1/2*np.pi* (ket01*ket10.dag() +ket10*ket01.dag()).full()
    psi_0 = qp.basis(2**num_qubits, 1).full()
    return psi_0, H

def other_entangled_hamiltonian():
    """
    Create an alternative entangled Hamiltonian for two qubits.

    Returns:
        np.ndarray: Hamiltonian matrix.
    """
    H = np.ones([4,4])
    H[0,3] = H[1,2] = H[2,1] = H[3,0] = -1.0
    print(H)
    return H


def result_string_to_dict(input_string:str)->dict[int,int]:
    """
    Convert a string of bits to a dictionary mapping index to bit value.

    Args:
        input_string (str): String of bits (e.g., '1010').

    Returns:
        dict: Dictionary mapping index to bit value.
    """
    return {i:int(bit) for i,bit in enumerate(list(input_string))}

def get_basepath():
    """
    Get base path for file access depending on current working directory.

    Returns:
        str: Base path string.
    """
    return '../' if os.getcwd()[-9:] == 'notebooks' else '' # for execution in jupyter notebooks

def get_velox_results(system: int)->pd.DataFrame:
    """
    Load Velox results for a given system from CSV and parse relevant fields.

    Args:
        system (int): System identifier.

    Returns:
        pd.DataFrame: DataFrame containing parsed results.
    """
    path = os.path.join(get_basepath(), f'data/results/hessian/{system}/best_results_hessian_{system}_native.csv')
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

def get_dwave_sample_set(system: int, timepoints: int, topology="1.4") -> pd.DataFrame:
    path = f'../data/results/hessian/{system}/{topology}'
    for file in os.listdir(path):
        file_tp = int(re.findall(r'(?<=timepoints_)\d+',file)[0])
        if not file_tp == timepoints:
            continue

        with open(os.path.join(path,file),'r') as f:
            s = dimod.SampleSet.from_serializable(json.load(f))
        
        if np.round(s.first.energy,12) == 0:
            return s
        print(s.first.energy)
    raise ValueError('No ground state for given system tp pair found')




def get_dwave_success_rates(system: int,topology="6.4",ta=200,grouped=True,file_limit=np.inf)->pd.DataFrame:
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
    path = f'../data/results/hessian/{system}/'
    dfs = []
    df_dict = defaultdict(list)
    for topology in [topology]:
        path += topology
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

def get_precision_timepoints_pairs(dfs):
    """
    Get all unique (precision, timepoints) pairs from a DataFrame.

    Args:
        dfs (pd.DataFrame): DataFrame containing 'precision' and 'timepoints' columns.

    Returns:
        list: List of unique (precision, timepoints) pairs.
    """
    dfs = dfs.groupby(['precision','timepoints'])['num_occurrences'].count()
    return list(set(dfs.index))


def return_tts(p_success: float,t:float, p_target=0.99)->float:
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


def get_velox_tts(system:int)->pd.DataFrame:
    """
    Compute Velox success rates for a given system.

    Args:
        system (int): System identifier.

    Returns:
        pd.DataFrame: DataFrame containing aggregated success rates and runtimes.
    """
    df = get_velox_results(system=system)
    df['tts99'] = df.apply(lambda row: return_tts(row['success_prob'],row.runtime),axis=1)
    df = df[['precision','timepoints','num_var','tts99']].groupby(['precision','timepoints','num_var']).min().reset_index()
    df['system'] = system
    df['source'] = 'VELOX'
    return df


def get_dwave_tts(system: int,topology="6.4",file_limit=np.inf)->pd.DataFrame:

    path = f'../data/results/hessian/{system}/'
    df_dict = defaultdict(list)
    path += topology
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
    df['tts99'] = df.apply(lambda row: return_tts(row['success_prob'],row.runtime),axis=1)
    df = df[['precision','timepoints','num_var','tts99']].groupby(['precision','timepoints','num_var']).min().reset_index()
    df['source'] = topology
    df['system'] = system
    return df
