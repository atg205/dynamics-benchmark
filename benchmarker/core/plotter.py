import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from pathlib import Path
from .results import BenchmarkResult
import matplotlib as mpl
from benchmarker.core import results_loader, instance
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm
import qutip as qp
import math
from ..config import PLOTS_DIR, ensure_dir


class BenchmarkPlotter:
    def __init__(self, output_dir: Optional[Path] = None,grid=True):
        """
        Initialize the BenchmarkPlotter.
        
        Args:
            output_dir: Optional custom output directory. If None, uses the default plots directory.
        """
        self.output_dir = output_dir if output_dir else PLOTS_DIR
        ensure_dir(self.output_dir)
        self._setup_style(grid=grid)
    
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
            save_path = self.output_dir / 'results_comparison.pdf'
            plt.savefig(save_path, bbox_inches='tight')
        plt.show(block=True)
        #plt.close()

    def plot_tts(self, systems: List[int] = [1,2,5,6,7], file_limit: int = 20, num_reps=0):
        """Plot time-to-solution comparisons for multiple systems."""

        ta = 200
        self._setup_style(fontsize=16, grid=True)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        colors = plt.cm.tab10.colors  # Use tab10 colormap for better consistency
        
        ALL_dfs = []
        loader = results_loader.ResultsLoader()

        for _, system in tqdm(enumerate(systems), total=len(systems)):
            velox_tts = loader.get_velox_tts(system)
            dwave_14_tts = loader.get_dwave_tts(system, topology='1.4', file_limit=file_limit, ta=200,num_reps=num_reps)
            dwave_64_tts = loader.get_dwave_tts(system, topology='6.4', file_limit=file_limit, ta=200,num_reps=num_reps)
            neal_tts = loader.get_dwave_tts(system, topology='neal', file_limit=file_limit, num_reps=num_reps)
            ALL_dfs.append(velox_tts)
            ALL_dfs.append(dwave_14_tts)
            ALL_dfs.append(dwave_64_tts)
            ALL_dfs.append(neal_tts)

        sources = ['VELOX', '1.4', '6.4','neal']
        linestyles = ['-', '-', '-','-']
        all_system_dfs = pd.concat(ALL_dfs, axis=0)
        native_system_df = all_system_dfs[all_system_dfs.system.isin(systems)]

        for i, source in enumerate(sources):
            native_system_df_filtered = native_system_df[native_system_df.source == source]
            
            # Group by num_var and calculate mean and std for tts99
            grouped = native_system_df_filtered.groupby('num_var')['tts99'].agg(['mean', 'std', 'count']).reset_index()
            print(grouped)
            # Filter out NaN/infinite values
            mask = np.isfinite(grouped['mean'])
            grouped_clean = grouped[mask]
            
            num_var_clean = np.array(grouped_clean['num_var'])
            tts99_mean = np.array(grouped_clean['mean'])
            tts99_std = np.array(grouped_clean['std'])
            
            # Handle cases where std is NaN (only one data point)
            tts99_std = np.where(np.isnan(tts99_std), 0, tts99_std)
            
            # Plot points
            ax.plot(
                num_var_clean,
                tts99_mean,
                marker=['o', 's', '^','x'][i],
                linestyle='',
                color=colors[i],
                label=['VeloxQ', 'Advantage2 1.4', 'Advantage 6.4','neal'][i],
                markersize=8,
                alpha=0.8,
            )
            
            # Fit exponential trend to averaged data
            if len(num_var_clean) > 1:
                log_tts99_mean = np.log(tts99_mean)
                slope, intercept, r_value, p_value, std_err = linregress(num_var_clean, log_tts99_mean)
                
                r_TTS99 = slope
                D = np.exp(intercept)
                
                # Generate smooth curve for fit
                num_var_fit = np.linspace(num_var_clean.min(), num_var_clean.max(), 100)
                TTS99_fit = D * np.exp(r_TTS99 * num_var_fit)
                
                ax.plot(
                    num_var_fit, TTS99_fit, 
                    linestyle=linestyles[i], 
                    color=colors[i], 
                    alpha=0.7
                )
                
                # Add equation annotation
                mid_x = 1.1 * (num_var_clean.min() + num_var_clean.max()) / 2
                mid_y = D * np.exp(r_TTS99 * mid_x)
                
                # Format equation text
                eq_text = rf"$\propto e^{{\frac{{N}}{{{1/r_TTS99:.1f}}}}}$" if abs(r_TTS99) > 0.001 else rf"$\propto \mathrm{{const}}$"
                
                ax.annotate(
                    eq_text,
                    xy=(mid_x, mid_y * [1.3, 1.2, 0.8,0.7][i]),
                    fontsize=24,
                    color=colors[i],
                    rotation=[0, 5, 30, 0][i],
                    ha='center',
                    va='bottom'
                )

        # Place legend inside the plot
        ax.legend(
            loc='upper right',
            ncol=1,
            fontsize=13,
            framealpha=1.0,
            handlelength=1
        )
        
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$\mathrm{TTS}_{\rm 99}$ [ms]")
        ax.set_yscale("log")
        #ax.set_ylim(1e1, 1e4)
        ax.grid(True)
        
        plt.tight_layout()  # No need for extra space since legend is inside
        plt.savefig(f"{self.output_dir}/tta_overview.pdf", bbox_inches="tight")
        plt.show(block=True)

    def plot_tts_horizontal_legend(self, systems: List[int] = [1,2,5,6,7], file_limit: int = 20, num_reps=0):
        """Plot time-to-solution comparisons for multiple systems with horizontal legend above."""
        self._setup_style(fontsize=16, grid=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = plt.cm.tab10.colors  # Use tab10 colormap for better consistency
        
        ALL_dfs = []
        loader = results_loader.ResultsLoader()

        for idx, system in tqdm(enumerate(systems), total=len(systems)):
            velox_tts = loader.get_velox_tts(system)
            dwave_14_tts = loader.get_dwave_tts(system, topology='1.4', file_limit=file_limit, num_reps=num_reps)
            dwave_64_tts = loader.get_dwave_tts(system, topology='6.4', file_limit=file_limit, num_reps=num_reps)

            ALL_dfs.append(velox_tts)
            ALL_dfs.append(dwave_14_tts)
            ALL_dfs.append(dwave_64_tts)

        sources = ['VELOX', '1.4', '6.4']
        linestyles = ['-', '-', '-']
        all_system_dfs = pd.concat(ALL_dfs, axis=0)
        native_system_df = all_system_dfs[all_system_dfs.system.isin(systems)]

        for i, source in enumerate(sources):
            native_system_df_filtered = native_system_df[native_system_df.source == source]
            
            # Group by num_var and calculate mean and std for tts99
            grouped = native_system_df_filtered.groupby('num_var')['tts99'].agg(['mean', 'std', 'count']).reset_index()
            
            # Filter out NaN/infinite values
            mask = np.isfinite(grouped['mean'])
            grouped_clean = grouped[mask]
            
            num_var_clean = np.array(grouped_clean['num_var'])
            tts99_mean = np.array(grouped_clean['mean'])
            tts99_std = np.array(grouped_clean['std'])
            
            # Handle cases where std is NaN (only one data point)
            tts99_std = np.where(np.isnan(tts99_std), 0, tts99_std)
            
            # Plot points
            ax.plot(
                num_var_clean,
                tts99_mean,
                marker=['o', 's', '^'][i],
                linestyle='',
                color=colors[i],
                label=['VeloxQ', 'Advantage2 1.4', 'Advantage 6.4'][i],
                markersize=8,
                alpha=0.8,
            )
            
            # Fit exponential trend to averaged data
            if len(num_var_clean) > 1:
                log_tts99_mean = np.log(tts99_mean)
                slope, intercept, r_value, p_value, std_err = linregress(num_var_clean, log_tts99_mean)
                
                r_TTS99 = slope
                D = np.exp(intercept)
                
                # Generate smooth curve for fit
                num_var_fit = np.linspace(num_var_clean.min(), num_var_clean.max(), 100)
                TTS99_fit = D * np.exp(r_TTS99 * num_var_fit)
                
                ax.plot(
                    num_var_fit, TTS99_fit, 
                    linestyle=linestyles[i], 
                    color=colors[i], 
                    alpha=0.7
                )
                
                # Add equation annotation
                mid_x = 1.1 * (num_var_clean.min() + num_var_clean.max()) / 2
                mid_y = D * np.exp(r_TTS99 * mid_x)
                
                # Format equation text
                eq_text = rf"$\propto e^{{{r_TTS99:.3f}  N}}$" if abs(r_TTS99) > 0.001 else rf"$\propto \mathrm{{const}}$"
                
                ax.annotate(
                    eq_text,
                    xy=(mid_x, mid_y * [1.3, 1.2, 0.8,0.7][i]),
                    fontsize=18,
                    color=colors[i],
                    rotation=[0, 5, 30][i],
                    ha='center',
                    va='bottom'
                )

        # Place legend above the plot in one row
        ax.legend(
            bbox_to_anchor=(0., 1.02, 1., .102),
            loc='lower center',
            ncol=3,  # Force all items in one row
            fontsize=13,
            mode="expand",
            borderaxespad=0.,
            handlelength=1
        )
        
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$\mathrm{TTS}_{\rm 99}$ [ms]")
        ax.set_yscale("log")
        ax.grid(True)
        
        plt.tight_layout(rect=(0, 0, 1, 0.90))  # Make room for the legend
        plt.savefig(f"{self.output_dir}/tta_overview_horizontal_legend.pdf", bbox_inches="tight")
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
