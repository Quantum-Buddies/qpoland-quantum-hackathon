"""
Ansatz diagrams for feasibility verification.

This script generates:
- Feature map (ZZ-like) diagram for n qubits
- Hardware-efficient ansatz (HEA) diagram with Ry + CZ ring
- CTQW Trotterization schematic e^{-i gamma A dt} repeated K times

Outputs are saved under results_wl/ansatz/
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def draw_wires(ax, n_qubits, length=10.0, y_spacing=1.2, y0=0.0):
    ys = []
    for i in range(n_qubits):
        y = y0 + i * y_spacing
        ys.append(y)
        ax.plot([0, length], [y, y], color='black', lw=1.5)
    return ys


def box(ax, x, y, w, h, label):
    rect = patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02", edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9)


def cz(ax, x, y1, y2):
    ax.plot([x, x], [y1, y2], color='black', lw=1.2)
    ax.plot([x], [y1], marker='o', color='black')
    ax.plot([x], [y2], marker='o', color='black')
    ax.text(x, (y1+y2)/2, 'CZ', ha='left', va='center', fontsize=7, color='#444')


def ring_pairs(n):
    pairs = [(i, (i + 1) % n) for i in range(n)]
    return pairs


def draw_feature_map(n_qubits=4, layers=1, save_path=Path('results_wl/ansatz/feature_map.png')):
    ensure_dir(save_path.parent)
    fig, ax = plt.subplots(figsize=(12, 1.2 * n_qubits + 1))
    ax.axis('off')

    length = 12
    ys = draw_wires(ax, n_qubits, length=length)

    x = 1.5
    for layer in range(layers):
        # Single-qubit encoders per qubit: RZ(x_i), RX(x_i)
        for q in range(n_qubits):
            box(ax, x, ys[q], 0.8, 0.5, 'RZ(x{})'.format(q))
            box(ax, x + 0.9, ys[q], 0.8, 0.5, 'RX(x{})'.format(q))
        x += 2.2

        # Entangling layer: CZ ring
        for (i, j) in ring_pairs(n_qubits):
            cz(ax, x, ys[i], ys[j])
        x += 1.4

    ax.text(0.2, ys[-1] + 0.7, 'ZZ-like Feature Map (schematic)', fontsize=12, ha='left', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def draw_hardware_efficient(n_qubits=4, layers=2, save_path=Path('results_wl/ansatz/hardware_efficient.png')):
    ensure_dir(save_path.parent)
    fig, ax = plt.subplots(figsize=(12, 1.2 * n_qubits + 1))
    ax.axis('off')

    length = 12
    ys = draw_wires(ax, n_qubits, length=length)

    x = 1.5
    for layer in range(layers):
        # Parameterized Ry on each qubit
        for q in range(n_qubits):
            box(ax, x, ys[q], 0.9, 0.5, 'Ry(θ{})'.format(layer * n_qubits + q))
        x += 1.4

        # CZ ring entanglers
        for (i, j) in ring_pairs(n_qubits):
            cz(ax, x, ys[i], ys[j])
        x += 1.4

    ax.text(0.2, ys[-1] + 0.7, 'Hardware-Efficient Ansatz (Ry + CZ ring)', fontsize=12, ha='left', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def draw_ctqw_trotter(K=4, save_path=Path('results_wl/ansatz/ctqw_trotter.png')):
    ensure_dir(save_path.parent)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    # Draw a time axis with repeated trotter blocks
    y = 1.5
    ax.plot([0.5, 11.5], [y, y], color='black', lw=1.5)
    ax.text(0.3, y, 't=0', ha='right', va='center')
    ax.text(11.7, y, 't=T', ha='left', va='center')

    x = 1.5
    for k in range(K):
        rect = patches.FancyBboxPatch((x - 0.8, y - 0.6), 1.6, 1.2, boxstyle="round,pad=0.02", edgecolor='black', facecolor='#e8f4fa')
        ax.add_patch(rect)
        ax.text(x, y + 0.2, r"e$^{-i\,\gamma\,A\,\Delta t}$", ha='center', va='center', fontsize=12)
        ax.text(x, y - 0.4, 'Trotter step {}'.format(k + 1), ha='center', va='center', fontsize=8)
        x += 2.2

    ax.text(0.6, 2.7, 'CTQW Trotterization schematic for H = -γ A', fontsize=12, ha='left')
    ax.text(0.6, 2.3, 'Each block approximates evolution by Δt; repeat K times: U(T) ≈ ∏ exp(-i γ A Δt)', fontsize=9, ha='left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    draw_feature_map(n_qubits=4, layers=1)
    draw_hardware_efficient(n_qubits=4, layers=2)
    draw_ctqw_trotter(K=5)
    print('Saved ansatz diagrams to results_wl/ansatz')
