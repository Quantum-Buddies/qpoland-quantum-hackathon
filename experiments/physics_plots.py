"""
Physics-level diagnostics for Continuous-Time Quantum Walk (CTQW) on molecular graphs.

Generates the following plots for a selected graph:
- Eigenvalue spectrum (Adjacency and Laplacian)
- Return probability vs time (average diagonal of |U(t)|^2)
- Shannon entropy vs time (of position probabilities)
- Quantum coherence (l1-norm of off-diagonals of density matrix) vs time
- Trace[U(t)] real/imag vs time
- Mixing matrix heatmaps |U(t)|^2 at representative times
- Time-averaged mixing matrix heatmap (approximate integral)

Outputs saved to results_wl/physics/
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.linalg import expm

# Local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from qkernels.datasets import MolecularGraphDataset


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def graph_matrices(G: nx.Graph):
    A = nx.to_numpy_array(G, dtype=float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    return A, L


def compute_ctqw_quantities(A: np.ndarray, gamma: float, times: np.ndarray):
    n = A.shape[0]
    H = -gamma * A

    # Uniform initial state
    psi0 = np.ones(n, dtype=complex) / np.sqrt(n)

    # Storage
    tr_real, tr_imag = [], []
    entropy = []
    coherence = []
    avg_return_prob = []
    mixing_mats = []

    # Precompute identity for trace if needed
    for t in times:
        U = expm(-1j * H * t)
        psi_t = U @ psi0
        probs = np.abs(psi_t) ** 2

        # Shannon entropy
        p = probs + 1e-16
        entropy.append(-(p * np.log2(p)).sum())

        # l1-coherence of pure state rho = |psi><psi|
        # C_l1 = sum_{i!=j} |rho_ij| = (sum |psi_i|)^2 - 1
        s = np.abs(psi_t).sum()
        coherence.append(s * s - 1.0)

        # Return probability (average of diagonal of |U|^2)
        M = np.abs(U) ** 2
        mixing_mats.append(M)
        avg_return_prob.append(np.mean(np.diag(M)))

        # Trace
        tr = np.trace(U)
        tr_real.append(np.real(tr))
        tr_imag.append(np.imag(tr))

    return {
        'tr_real': np.array(tr_real),
        'tr_imag': np.array(tr_imag),
        'entropy': np.array(entropy),
        'coherence': np.array(coherence),
        'avg_return_prob': np.array(avg_return_prob),
        'mixing_mats': mixing_mats,
    }


def plot_spectrum(A: np.ndarray, L: np.ndarray, out_dir: Path):
    wA = np.linalg.eigvalsh(A)
    wL = np.linalg.eigvalsh(L)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title('Adjacency spectrum')
    sns.histplot(wA, bins=20, kde=False, color='#1f77b4')
    plt.xlabel('Eigenvalue')

    plt.subplot(1, 2, 2)
    plt.title('Laplacian spectrum')
    sns.histplot(wL, bins=20, kde=False, color='#d62728')
    plt.xlabel('Eigenvalue')

    plt.tight_layout()
    plt.savefig(out_dir / 'spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_timeseries(times: np.ndarray, data: np.ndarray, title: str, ylabel: str, out_file: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(times, data, lw=2)
    plt.title(title)
    plt.xlabel('time t')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mixing_heatmaps(times: np.ndarray, mixing_mats: list, out_dir: Path):
    # Pick a few representative times
    idxs = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    plt.figure(figsize=(14, 3))
    for i, idx in enumerate(idxs, 1):
        plt.subplot(1, len(idxs), i)
        sns.heatmap(mixing_mats[idx], cmap='viridis', cbar=(i == len(idxs)))
        plt.title(f'|U(t)|^2, t={times[idx]:.2f}')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / 'mixing_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Time-averaged mixing matrix
    avg_M = np.mean(np.stack(mixing_mats, axis=0), axis=0)
    plt.figure(figsize=(5, 4))
    sns.heatmap(avg_M, cmap='viridis', cbar=True)
    plt.title('Time-averaged mixing matrix')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / 'mixing_avg.png', dpi=300, bbox_inches='tight')
    plt.close()


def main(dataset='MUTAG', graph_index=0, gamma=1.0, T=6.0, steps=200):
    out_dir = Path('results_wl') / 'physics' / dataset
    ensure_dir(out_dir)

    # Load dataset and pick a graph
    ds = MolecularGraphDataset(dataset, data_dir='data')
    G = ds.graphs[graph_index]

    A, L = graph_matrices(G)
    times = np.linspace(0.0, T, steps)

    # Compute CTQW diagnostics
    ctqw = compute_ctqw_quantities(A, gamma, times)

    # Plots
    plot_spectrum(A, L, out_dir)
    plot_timeseries(times, ctqw['avg_return_prob'], 'Average return probability', 'Avg return prob', out_dir / 'avg_return_prob.png')
    plot_timeseries(times, ctqw['entropy'], 'Shannon entropy of position', 'Entropy (bits)', out_dir / 'entropy.png')
    plot_timeseries(times, ctqw['coherence'], 'Quantum coherence (l1)', 'Coherence', out_dir / 'coherence.png')
    plot_timeseries(times, ctqw['tr_real'], 'Trace[U(t)] real part', 'Re Tr(U)', out_dir / 'trace_real.png')
    plot_timeseries(times, ctqw['tr_imag'], 'Trace[U(t)] imag part', 'Im Tr(U)', out_dir / 'trace_imag.png')
    plot_mixing_heatmaps(times, ctqw['mixing_mats'], out_dir)

    # Save quick summary text
    with open(out_dir / 'SUMMARY.txt', 'w') as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Graph index: {graph_index}\n")
        f.write(f"Nodes: {A.shape[0]}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Time range: [0, {T}] with {steps} steps\n")

    print(f"Saved physics plots to {out_dir}")


if __name__ == '__main__':
    main()
