import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import os
from .results import BenchmarkResult
import matplotlib as mpl
from benchmarker.core import results_loader
import pandas as pd
from scipy.stats import linregress


class BenchmarkPlotter:
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._setup_style()
    
    def _setup_style(self,fontsize=15,scale=1,grid=True):
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
    
    def plot_dynamics(self, result: BenchmarkResult, 
                     baseline: Optional[List[float]] = None,
                     save: bool = True) -> None:
        """Plot dynamics results with optional baseline comparison"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot result data
        times = result.result['time'].values
        expectation = result.result['expectation'].values
        ax.scatter(times, expectation, marker='o', label='Quantum Annealer')
        
        if baseline is not None:
            ax.plot(times, baseline, 'k--', label='Classical (baseline)')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$\langle \sigma_z \rangle$')
        ax.set_title(f'System {result.system} (ta={result.ta})')
        ax.legend()
        
        if save:
            plt.savefig(os.path.join(
                self.output_dir, 
                f'dynamics_system_{result.system}_ta_{result.ta}.pdf'
            ), bbox_inches='tight')
        plt.close()
    
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

    def plot_tts(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.tab10.colors  # do 10 systemów
        systems = [i for i in range(1,10)]
        systems = [2,3,4,6,8]
        ALL_dfs = []
        native_systems = [1,2,5,6,7]
        loader = results_loader.ResultsLoader()

        for idx, system in enumerate(native_systems):
            velox_tts = loader.get_velox_tts(system)
            dwave_14_tts = loader.get_dwave_tts(system,topology='1.4',file_limit=20)
            dwave_64_tts = loader.get_dwave_tts(system,topology='6.4',file_limit=20)

            color = colors[idx % len(colors)]

            print("--------------")
            print(f"System {system}")
            #print(velox_tts)
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
        native_system_df = all_system_dfs[all_system_dfs.system.isin(native_systems)]

        for i,source in enumerate(sources):

            native_system_df_filtered = native_system_df[native_system_df.source == source]
            num_var = np.array(native_system_df_filtered['num_var'])       
            TTS99 = np.array(native_system_df_filtered['tts99'])           

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
        ax.grid(True)
        plt.tight_layout()
        #plt.savefig(f'../plots/tta_overview.pdf' ,bbox_inches='tight')
        #ax.set_ylim(0,1e6)
        ylims = ax.get_ylim()

        plt.show(block=True)