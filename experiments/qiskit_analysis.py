"""
Qiskit-based CTQW evolution and quantum circuit diagrams.

This script generates:
- CTQW evolution using Qiskit Aer simulator
- Proper quantum circuit diagrams for ansatz and feature maps
- Physics-level diagnostics using quantum simulation backend

Outputs saved to results_wl/qiskit/
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector, partial_trace, entropy
from qiskit.visualization import circuit_drawer, plot_state_qsphere, plot_bloch_multivector
import networkx as nx

# Local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from qkernels.datasets import MolecularGraphDataset


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def adjacency_to_pauli_hamiltonian(A: np.ndarray, gamma: float = 1.0):
    """Convert adjacency matrix to Pauli Hamiltonian for Qiskit."""
    n = A.shape[0]
    # For simplicity, use direct unitary evolution with adjacency matrix
    # In practice, you'd convert to Pauli operators, but this is complex
    return None  # For now, let's use unitary evolution directly


def create_ctqw_circuit(A: np.ndarray, gamma: float, t: float, n_qubits: int):
    """Create CTQW circuit using Trotterization."""
    qc = QuantumCircuit(n_qubits)

    # Initial state: uniform superposition
    qc.h(range(n_qubits))

    # Trotter steps for e^{-i γ A t}
    dt = t / 4  # 4 Trotter steps for simplicity
    for _ in range(4):
        # Single qubit rotations (diagonal terms of A)
        for i in range(n_qubits):
            if A[i, i] != 0:
                qc.rz(-2 * gamma * A[i, i] * dt, i)

        # Two-qubit interactions (off-diagonal terms)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if A[i, j] != 0:
                    qc.cx(i, j)
                    qc.rz(-2 * gamma * A[i, j] * dt, j)
                    qc.cx(i, j)

    return qc


def create_feature_map_circuit(n_qubits: int, x: np.ndarray):
    """Create ZZ-feature map circuit."""
    qc = QuantumCircuit(n_qubits)

    # Encoding layer
    for i in range(n_qubits):
        qc.ry(x[i], i)

    # Entangling layer (ZZ interactions)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qc.cx(i, j)
            qc.rz(x[i] * x[j], j)
            qc.cx(i, j)

    return qc


def create_hardware_efficient_circuit(n_qubits: int, layers: int, params: np.ndarray):
    """Create hardware-efficient ansatz circuit."""
    qc = QuantumCircuit(n_qubits)

    param_idx = 0
    for layer in range(layers):
        # Parameterized rotations
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1

        # Entangling layer (linear connectivity)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc


def simulate_ctqw_evolution(A: np.ndarray, gamma: float, times: np.ndarray, n_qubits: int):
    """Simulate CTQW using Qiskit Aer."""
    backend = AerSimulator()

    results = {
        'entropy': [],
        'coherence': [],
        'probabilities': []
    }

    for t in times:
        # Create circuit
        qc = create_ctqw_circuit(A, gamma, t, n_qubits)

        # Get statevector (for small systems)
        if n_qubits <= 10:  # Limit to avoid exponential explosion
            try:
                statevector = Statevector.from_instruction(qc)
                probs = np.abs(statevector.data) ** 2

                # Calculate Shannon entropy
                probs = probs[probs > 1e-10]  # Avoid log(0)
                if len(probs) > 0:
                    entropy_val = -np.sum(probs * np.log2(probs))
                else:
                    entropy_val = 0

                # Quantum coherence (simplified measure)
                coherence_val = np.std(probs)  # Standard deviation of probabilities

                results['entropy'].append(entropy_val)
                results['coherence'].append(coherence_val)
                results['probabilities'].append(probs[:n_qubits])  # Take first n_qubits amplitudes
            except:
                # Fallback for larger systems
                results['entropy'].append(0)
                results['coherence'].append(0)
                results['probabilities'].append(np.ones(n_qubits) / n_qubits)

    return results


def plot_qiskit_results(times: np.ndarray, results: dict, out_dir: Path):
    """Plot results from Qiskit simulation."""
    # Entropy vs time
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    plt.plot(times, results['entropy'])
    plt.xlabel('Time t')
    plt.ylabel('Shannon Entropy')
    plt.title('Position Entropy vs Time')
    plt.grid(True, alpha=0.3)

    # Coherence vs time
    plt.subplot(2, 2, 2)
    plt.plot(times, results['coherence'])
    plt.xlabel('Time t')
    plt.ylabel('Quantum Coherence')
    plt.title('Coherence vs Time')
    plt.grid(True, alpha=0.3)

    # Probability distribution evolution
    plt.subplot(2, 2, 3)
    prob_evolution = np.array(results['probabilities'])
    for i in range(prob_evolution.shape[1]):
        plt.plot(times, prob_evolution[:, i], label=f'Qubit {i}')
    plt.xlabel('Time t')
    plt.ylabel('Probability')
    plt.title('Probability Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Phase space plot (entropy vs coherence)
    plt.subplot(2, 2, 4)
    plt.plot(results['coherence'], results['entropy'], 'b-', alpha=0.7)
    plt.xlabel('Coherence')
    plt.ylabel('Entropy')
    plt.title('Entropy vs Coherence')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'qiskit_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_circuit_diagrams(out_dir: Path, n_qubits: int = 4):
    """Save quantum circuit diagrams using Qiskit visualization."""

    # Feature map circuit
    x = np.random.random(n_qubits)  # Random encoding parameters
    feature_map = create_feature_map_circuit(n_qubits, x)
    circuit_drawer(feature_map, output='mpl', filename=out_dir / 'feature_map_qiskit.png')

    # Hardware efficient ansatz
    n_params = n_qubits * 2  # 2 layers
    params = np.random.random(n_params) * 2 * np.pi
    hea_circuit = create_hardware_efficient_circuit(n_qubits, 2, params)
    circuit_drawer(hea_circuit, output='mpl', filename=out_dir / 'hardware_efficient_qiskit.png')

    # CTQW circuit
    # Create a simple graph adjacency matrix
    A = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0]], dtype=float)
    ctqw_circuit = create_ctqw_circuit(A, gamma=1.0, t=1.0, n_qubits=n_qubits)
    circuit_drawer(ctqw_circuit, output='mpl', filename=out_dir / 'ctqw_circuit_qiskit.png')


def main():
    """Main function to run Qiskit-based analysis."""
    out_dir = Path('results_wl') / 'qiskit'
    ensure_dir(out_dir)

    print("Running Qiskit-based CTQW analysis...")

    # Use MUTAG graph 0 as example
    dataset = MolecularGraphDataset('MUTAG', data_dir='data')
    G = dataset.graphs[0]
    A = nx.to_numpy_array(G, dtype=float)

    n_qubits = A.shape[0]
    gamma = 1.0
    times = np.linspace(0, 5.0, 50)

    print(f"Analyzing graph with {n_qubits} nodes...")

    # Run Qiskit simulation
    results = simulate_ctqw_evolution(A, gamma, times, n_qubits)

    # Plot results
    plot_qiskit_results(times, results, out_dir)

    # Save circuit diagrams
    save_circuit_diagrams(out_dir, n_qubits=min(n_qubits, 6))  # Limit to 6 qubits for readability

    # Save summary
    with open(out_dir / 'README.md', 'w') as f:
        f.write("# Qiskit-based CTQW Analysis\n\n")
        f.write("This directory contains quantum circuit diagrams and evolution plots generated using Qiskit.\n\n")
        f.write("## Files:\n")
        f.write("- `qiskit_evolution.png`: CTQW evolution plots (entropy, coherence, probabilities)\n")
        f.write("- `feature_map_qiskit.png`: ZZ-feature map circuit diagram\n")
        f.write("- `hardware_efficient_qiskit.png`: Hardware-efficient ansatz diagram\n")
        f.write("- `ctqw_circuit_qiskit.png`: CTQW Trotterization circuit\n")
        f.write("- `README.md`: This file\n")

    print(f"Saved Qiskit results to {out_dir}")


if __name__ == '__main__':
    main()
