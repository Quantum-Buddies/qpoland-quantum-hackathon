# Qiskit Circuit Diagrams

This directory contains proper quantum circuit diagrams generated using Qiskit.

## Circuits Generated:

### 1. ZZ-Feature Map
- **Purpose**: Encoding classical data into quantum states
- **Structure**: Ry rotations followed by ZZ entangling gates
- **Parameters**: Feature vector x ∈ ℝⁿ
- **Depth**: O(n²) due to pairwise entangling gates

### 2. Hardware-Efficient Ansatz (HEA)
- **Purpose**: Variational quantum circuit for optimization
- **Structure**: Layers of Ry rotations + linear connectivity CNOTs
- **Parameters**: θ ∈ ℝ^(n×layers)
- **Depth**: O(n×layers)
- **Connectivity**: Linear (nearest-neighbor)

### 3. CTQW Trotterization
- **Purpose**: Simulating continuous-time quantum walk evolution
- **Structure**: Trotter steps approximating e^{-iγAt}
- **Components**: Diagonal rotations + pairwise CNOT-RZ-CNOT
- **Parameters**: γ (coupling), t (time), Δt (Trotter step size)
- **Approximation**: First-order Trotter-Suzuki

## Implementation Details

### CTQW Hamiltonian
H = -γA where A is the adjacency matrix
- **Diagonal terms**: Single-qubit Z rotations
- **Off-diagonal terms**: Two-qubit ZZ interactions via CNOT gates
- **Initial state**: Uniform superposition |+⟩^⊗n

### Circuit Identities
- **Trotter step**: e^{-iγAΔt} ≈ ∏e^{-iγA_ij Δt}
- **Two-qubit evolution**: e^{-iγΔt σᶻ⊗σᶻ} = CNOT-RZ(-2γΔt)-CNOT
- **Time evolution**: U(t) = [e^{-iγAΔt}]^K with KΔt = t

## Files:
- `feature_map.png`: ZZ-feature map circuit diagram
- `hardware_efficient.png`: Hardware-efficient ansatz diagram
- `ctqw_trotter.png`: CTQW Trotterization circuit
- `README.md`: This documentation
