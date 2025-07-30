import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import os
from .results import BenchmarkResult
import matplotlib as mpl


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