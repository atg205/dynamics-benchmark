import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import os
from .results import BenchmarkResult
import matplotlib as mpl
from benchmarker.core import results_loader, instance
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm
import qutip as qp
import math

class BenchmarkPlotter:
    def __init__(self, output_dir: str = "benchmarker/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._setup_style()
    
    def _setup_style(self,fontsize=15,scale=1.0,grid=True):
# Publication-quality (TikZ-like) matplotlib style setup
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
    
    def plot_comparison(self, results: List[BenchmarkResult], 
                       labels: Optional[List[str]] = None,
                       save: bool = True) -> None:
        """Plot comparison of multiple results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, result in enumerate(results):
            label = labels[i] if labels else f'Result {i}'
            ax.scatter( 
                result.result['time'].values,
                result.result['expectation'].values,
                label=label
            )
        
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$\langle \sigma_z \rangle$')
        ax.set_title('Results Comparison')
        ax.legend()
        
        if save:
            plt.savefig(os.path.join(
                self.output_dir,
                'results_comparison.pdf'
            ), bbox_inches='tight')
        plt.show(block=True)
        #plt.close()

    def plot_tts(self,systems=[1,2,5,6,7],file_limit=20):
        self._setup_style(grid=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.tab10.colors  # do 10 systemów
        ALL_dfs = []
        loader = results_loader.ResultsLoader()

        for idx, system in tqdm(enumerate(systems),total=len(systems)):
            velox_tts = loader.get_velox_tts(system)
            dwave_14_tts = loader.get_dwave_tts(system,topology='1.4',file_limit=file_limit)
            dwave_64_tts = loader.get_dwave_tts(system,topology='6.4',file_limit=file_limit)

            color = colors[idx % len(colors)]
            ax.plot(velox_tts['num_var'], velox_tts['tts99'],
                    marker='o', linestyle='None', color=colors[0], label=f'velox' if system==0 else None)
            
            ax.plot(dwave_14_tts['num_var'], dwave_14_tts['tts99'],
                    marker='s', linestyle='None', color=colors[1],label='14' if system==0 else None)
            
            ax.plot(dwave_64_tts['num_var'], dwave_64_tts['tts99'],
                    marker='^', linestyle='None', color=colors[2],label='64' if system==0 else None)
            ALL_dfs.append(velox_tts)
            ALL_dfs.append(dwave_14_tts)
            ALL_dfs.append(dwave_64_tts)

        sources = ['VELOX','1.4', '6.4']
        linestyles = ['-','-','-']
        all_system_dfs = pd.concat(ALL_dfs,axis=0)


        for i,source in enumerate(sources):

            df_filtered = all_system_dfs[all_system_dfs.source == source]
            num_var = np.array(df_filtered['num_var'])       
            TTS99 = np.array(df_filtered['tts99'])           

            mask = np.isfinite(TTS99)
            num_var_clean = num_var[mask]
            TTS99_clean = TTS99[mask]
            log_TTS99 = np.log(TTS99_clean)

            slope, intercept, r_value, p_value, std_err = linregress(num_var_clean, log_TTS99)

            r_TTS99 = slope
            D =np.exp(intercept)
            t = D * np.exp(r_TTS99 * num_var)

            TTS99_fit = D * np.exp(r_TTS99 * num_var)
            ax.plot(num_var, TTS99_fit, linestyle=linestyles[i], color=colors[i], label=f'{source} r_TTS99={abs(np.round(r_TTS99,2))}')    

        plt.legend()
        ax.set_xlabel(r'$N$')
        ax.set_ylabel('TTS99 [ms]')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/tta_overview.pdf' ,bbox_inches='tight')
        plt.show(block=True)

    def plot_success_prob_by_ta(self, system:int,annealing_times=[10,100,200,500],timepoints_of_interest=[2,3]):

        self._setup_style(fontsize=13)
        loader = results_loader.ResultsLoader()
        timepoints_of_interest = [2, 3]
        topologies = ['1.4', '6.4']
        solver_names = {
            '1.4':'Advantage2',
            '6.4':'Advantage',
        }
        systems = [1,3]

        # przygotowanie figure z dwoma subplotami
        colors = {2: 'tab:blue', 3: 'tab:orange'}
        linestyles = {'1.4': 'dashed', '6.4': 'solid'}
        markers = {'1.4': 'o', '6.4': '^'}
        titles = {2: rf'$\left| \Psi_{system} \right\rangle$', 9: '^'}


        for system in  systems:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            data = {tp: {t: [] for t in timepoints_of_interest} for tp in topologies}

            for topology in topologies:
                for ta in annealing_times:
                    df = loader.get_dwave_success_rates(system, topology=topology, ta=ta, grouped=True,file_limit=20)
                    df = df[df['timepoints'].isin(timepoints_of_interest)]
                    for tp in timepoints_of_interest:
                        val = df[df['timepoints'] == tp]['success_prob'].values
                        data[topology][tp].append(val[0] if len(val) > 0 else None)

            for topology in topologies:
                for tp in timepoints_of_interest:
                    tas = [ta for i,ta in enumerate(annealing_times) if data[topology][tp][i] >0 ]
                    data[topology][tp] = [p for p in data[topology][tp] if p >0]
                    ax.plot(
                        tas,
                        data[topology][tp],
                        label=f'{solver_names[topology]}, {tp} timepoints',
                        color=colors[tp],
                        linestyle=linestyles[topology],
                        marker=markers[topology]
                    )

            #ax.set_title(rf'$\widehat K_{system}$')
            ax.set_xlabel(r'Annealing Time [$\mu$s]')
            ax.set_yscale('log')
            ax.grid(True)

            ax.set_ylabel('Success probability')
            if system ==1:
                ax.legend(loc='lower left')
            ax.set_ylim(5e-5,1)

            plt.tight_layout()
            
            plt.savefig(f'{self.output_dir}/success_prob_by_ta_system_{system}.pdf',bbox_inches='tight')
            plt.show()


    def plot_dynamics(self, system,timepoints=3):

        self._setup_style(fontsize=16,grid=False)

        loader = results_loader.ResultsLoader()

        SZ = np.array([[1, 0], [0, -1]])
        colors = ["r", "g", "b", "c", "m", "y", "k"]
        markers = ["o", "s", "^", "D", "v", "<", ">"]
        for j,system in enumerate([system]):
            fig, ax = plt.subplots(1,1,figsize=(4, 4))

            i = instance.BenchmarkInstance(system,number_time_points=timepoints)

            problem = i.problem
            times = np.linspace(0, len(problem.times)-1, 100)
            
            H = i.H
            dim = int(math.log2(H.shape[0]))
            P_00 = qp.tensor([qp.sigmaz()]+ [qp.qeye(2)]*(dim-1))
            P_11 = qp.tensor([qp.qeye(2)]*(dim-1) + [qp.sigmaz()])
            H_qp = qp.Qobj(H,dims=[[2]*dim,[2]*dim])
            psi_0 = qp.Qobj(i.psi0)
            psi_0.dims = [[2]*dim, [1]]  # Naprawa błędu wymiarów

            baseline = qp.mesolve(H_qp, psi_0, times, e_ops=[P_00, P_11]).expect

            ax.plot(times, baseline[0], "k--")
            if dim== 2:
                ax.plot(times, baseline[1], "k--")

            dw_result = loader.get_dwave_sample_set(system,timepoints=timepoints)

            for idx,sample in enumerate(list(dw_result.samples(3))[::-1]):
                dw_vec = problem.interpret_sample(sample)
                energy = list(dw_result.to_pandas_dataframe()[0:3][::-1]['energy'])[idx]
                expect_00 = [(state.conj() @ P_00.full() @ state).real for state in dw_vec]
                expect_11 = [(state.conj() @ P_11.full() @ state).real for state in dw_vec]

                #axis.scatter(inst_obj.problem.times, exact_expect, marker="^", lw=2, s=300, edgecolors="b", facecolors="none", label="Exact solver")
                #axis.scatter(problem.times, sa_expect, marker="o", lw=2, s=100, edgecolors="r", facecolors="none", label="SA sampler")
                ax.scatter(problem.times, expect_00,color=colors[idx % len(colors)], marker=markers[idx % len(markers)],label=f"Energy: {abs(energy):.2f}",s=70)
                ax.plot(problem.times, expect_00, color=colors[idx % len(colors)], alpha=0.5, linewidth=0.3)
                if dim ==2:

                    ax.scatter(problem.times, expect_11,color=colors[idx % len(colors)], marker=markers[idx % len(markers)],s=70)
                    ax.plot(problem.times, expect_11, color=colors[idx % len(colors)], alpha=0.5, linewidth=0.3)
                ax.set_xlabel("t")
                ax.legend(loc='upper left',fontsize=14)


                ax.set_ylabel(r"$\langle \sigma_z \rangle$")
            plt.ylim(-4,4)
            plt.tight_layout(w_pad=2)
            plt.savefig(f'{self.output_dir}/dynamics_system_{system}_list.pdf' ,bbox_inches='tight')
            plt.show(block=True)





#    P_00 = qp.tensor([qp.sigmaz()]+ [qp.qeye(2)]*(dim-1))
 #   P_11 = qp.tensor([qp.qeye(2)]*(dim-1) + [qp.sigmaz()])



    # psi_0 = qp.Qobj(psi0)
    # psi_0.dims = [[2]*dim, [1]]  # Naprawa błędu wymiarów

    # # Parametry i obliczenia
    # times = np.linspace(0, num_time_points, 100)  
    # baseline = qp.mesolve(H_qp, psi_0, times, e_ops=[P_00, P_11]).expect

    # expect_00 = [(state.conj() @ P_00.full() @ state).real for state in dw_vec]
    # expect_11 = [(state.conj() @ P_11.full() @ state).real for state in dw_vec]
    # # Tworzenie wykresu Plotly
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=times, y=baseline[0], mode='lines', name='QuTiP (baseline)', line=dict(dash='dash', color='black')))

    # fig.add_trace(go.Scatter(x=[i for i in range(num_time_points)], y=expect_00, mode="markers", marker=dict(symbol="square", size=20, line=dict(width=2, color="green"), color="rgba(0,0,0,0)"), name="D-Wave sampler"))

    # if dim== 2:
    #     fig.add_trace(go.Scatter(x=times, y=baseline[1], mode='lines', name='QuTiP (baseline)', line=dict(dash='dash', color='blue')))
    #     fig.add_trace(go.Scatter(x=[i for i in range(num_time_points)], y=expect_11, mode="markers", marker=dict(symbol="circle", size=20, line=dict(width=2, color="red"), color="rgba(0,0,0,0)"), name="D-Wave sampler"))
