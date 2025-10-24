"""
Enhanced visualization utilities for QPoland Quantum Hackathon Challenge.

Includes:
1. Kernel matrix heatmaps for all methods (classical and quantum)
2. t-SNE embeddings of feature spaces
3. Decision boundary visualizations
4. Performance comparison plots
5. Quantum vs classical method comparisons
6. Feature importance analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def plot_quantum_vs_classical_comparison(results, save_dir='comprehensive_results'):
    """
    Plot comprehensive comparison between quantum and classical methods.

    Args:
        results: Dictionary of results from comprehensive benchmark
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Extract data for plotting
    plot_data = []

    for dataset_name, dataset_results in results.items():
        for method_name, method_results in dataset_results.items():
            plot_data.append({
                'Dataset': dataset_name,
                'Method': method_name,
                'Type': method_results.get('method_type', 'unknown'),
                'Accuracy': method_results['accuracy_mean'],
                'Accuracy_Std': method_results['accuracy_std'],
                'F1': method_results['f1_mean'],
                'F1_Std': method_results['f1_std'],
                'Best_C': method_results.get('best_C', 1.0),
                'Features': method_results.get('n_features', 'N/A')
            })

    df = pd.DataFrame(plot_data)

    # Separate quantum and classical methods
    quantum_methods = df[df['Type'] == 'quantum']
    classical_methods = df[df['Type'] != 'quantum']

    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Accuracy comparison
    datasets = df['Dataset'].unique()

    quantum_accs = []
    classical_accs = []

    for dataset in datasets:
        q_acc = quantum_methods[quantum_methods['Dataset'] == dataset]['Accuracy'].max()
        c_acc = classical_methods[classical_methods['Dataset'] == dataset]['Accuracy'].max()

        quantum_accs.append(q_acc if not np.isnan(q_acc) else 0)
        classical_accs.append(c_acc if not np.isnan(c_acc) else 0)

    x_pos = np.arange(len(datasets))
    width = 0.35

    ax1.bar(x_pos - width/2, quantum_accs, width, label='Quantum Methods', color='#ff6b6b', alpha=0.8)
    ax1.bar(x_pos + width/2, classical_accs, width, label='Classical Methods', color='#4ecdc4', alpha=0.8)

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Best Accuracy')
    ax1.set_title('Quantum vs Classical Performance Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Method type distribution
    type_counts = df['Type'].value_counts()
    colors = ['#4ecdc4', '#ff6b6b', '#45b7d1', '#96ceb4']
    ax2.pie(type_counts.values, labels=type_counts.index, colors=colors, autopct='%1.1f%%')
    ax2.set_title('Method Type Distribution')

    # 3. Accuracy box plot by type
    accuracy_by_type = []
    type_labels = []

    for method_type in df['Type'].unique():
        accuracies = df[df['Type'] == method_type]['Accuracy'].dropna()
        if len(accuracies) > 0:
            accuracy_by_type.append(accuracies)
            type_labels.append(method_type)

    if accuracy_by_type:
        bp = ax3.boxplot(accuracy_by_type, labels=type_labels, patch_artist=True)

        # Color boxes by type
        colors = ['#4ecdc4', '#ff6b6b', '#45b7d1', '#96ceb4']
        for patch, color in zip(bp['boxes'], colors[:len(accuracy_by_type)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Distribution by Method Type')
        ax3.grid(True, alpha=0.3)

    # 4. Feature count vs performance scatter
    valid_data = df[df['Features'] != 'N/A']
    if len(valid_data) > 0:
        scatter = ax4.scatter(valid_data['Features'], valid_data['Accuracy'],
                            c=valid_data['Accuracy'], cmap='viridis', s=100, alpha=0.7)

        # Add method type colors
        for _, row in valid_data.iterrows():
            if row['Type'] == 'quantum':
                ax4.scatter(row['Features'], row['Accuracy'], c='#ff6b6b', s=150,
                          marker='*', edgecolors='black', linewidth=2, alpha=0.8)
            else:
                ax4.scatter(row['Features'], row['Accuracy'], c='#4ecdc4', s=100,
                          marker='o', alpha=0.7)

        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Feature Count vs Performance')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Accuracy')

    plt.tight_layout()
    plt.savefig(save_dir / 'quantum_vs_classical_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {save_dir / 'quantum_vs_classical_comparison.png'}")

    return fig


def plot_kernel_matrices_comparison(results, dataset_name='MUTAG', save_dir='comprehensive_results'):
    """
    Plot kernel matrices for different methods to show learned structure.

    Args:
        results: Results dictionary
        dataset_name: Dataset to visualize
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    if dataset_name not in results:
        logger.warning(f"Dataset {dataset_name} not found in results")
        return

    dataset_results = results[dataset_name]

    # Load dataset
    from qkernels.datasets import MolecularGraphDataset
    dataset = MolecularGraphDataset(dataset_name)
    graphs, labels = dataset.get_graphs_and_labels()

    # Methods to visualize
    methods_to_plot = {
        'WL_Classical': 'Weisfeiler-Lehman (Classical)',
        'Topological_RBF': 'Topological + RBF',
        'Quantum_Fidelity': 'Quantum Fidelity Kernel',
        'Quantum_Walk': 'Quantum Walk Embedding'
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i, (method_key, method_name) in enumerate(methods_to_plot.items()):
        if method_key not in dataset_results:
            logger.warning(f"Method {method_key} not found for {dataset_name}")
            continue

        ax = axes[i]

        try:
            # Compute kernel matrix for this method
            if method_key == 'WL_Classical':
                from qkernels.classical_baselines import WeisfeilerLehmanKernel
                kernel = WeisfeilerLehmanKernel(h=3)
                K = kernel.fit(graphs).transform(graphs)
            elif method_key == 'Topological_RBF':
                from qkernels.features import TopologicalFeatureExtractor
                from sklearn.metrics.pairwise import rbf_kernel
                extractor = TopologicalFeatureExtractor()
                scaler = extractor.fit_scaler(graphs)
                features = np.array([extractor.extract_features(g) for g in graphs])
                features_scaled = scaler.transform(features)
                K = rbf_kernel(features_scaled, gamma=1.0)
            elif method_key == 'Quantum_Fidelity':
                from qkernels.quantum import create_fidelity_kernel_matrix
                features, _ = extract_features_for_quantum(graphs)
                K = create_fidelity_kernel_matrix(features, n_qubits=4, n_layers=1)
            elif method_key == 'Quantum_Walk':
                from qkernels.quantum import create_quantum_walk_kernel_matrix
                K = create_quantum_walk_kernel_matrix(graphs, n_qubits=4, n_layers=1)

            # Plot kernel matrix
            # Sort by labels for better visualization
            sort_idx = np.argsort(labels)
            K_sorted = K[sort_idx][:, sort_idx]

            im = ax.imshow(K_sorted, cmap='viridis', aspect='auto')

            # Add class boundaries
            unique_labels = np.unique(labels)
            boundaries = []
            for label in unique_labels[:-1]:
                boundary = np.sum(labels[sort_idx] == label)
                boundaries.append(boundary)

            for boundary in boundaries:
                ax.axhline(y=boundary - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
                ax.axvline(x=boundary - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)

            ax.set_title(f'{method_name}\nKernel Matrix', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Sample Index')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        except Exception as e:
            logger.error(f"Failed to plot {method_key}: {e}")
            ax.text(0.5, 0.5, f'Failed to\ncompute\n{e}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{method_name} (Failed)')

    plt.tight_layout()
    plt.savefig(save_dir / f'{dataset_name}_kernel_matrices_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Kernel matrices saved to {save_dir / f'{dataset_name}_kernel_matrices_comparison.png'}")

    return fig


def extract_features_for_quantum(graphs):
    """Extract features suitable for quantum processing."""
    from qkernels.features import HybridFeatureExtractor
    extractor = HybridFeatureExtractor()
    scaler = extractor.fit_scaler(graphs)
    features = np.array([extractor.extract_features(g) for g in graphs])
    features_scaled = scaler.transform(features)
    return features_scaled, extractor.get_feature_names()


def plot_feature_space_analysis(results, dataset_name='MUTAG', save_dir='comprehensive_results'):
    """
    Analyze and visualize feature spaces using t-SNE and PCA.

    Args:
        results: Results dictionary
        dataset_name: Dataset to analyze
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    if dataset_name not in results:
        logger.warning(f"Dataset {dataset_name} not found in results")
        return

    # Load dataset
    from qkernels.datasets import MolecularGraphDataset
    dataset = MolecularGraphDataset(dataset_name)
    graphs, labels = dataset.get_graphs_and_labels()

    # Feature extraction methods to compare
    feature_methods = {
        'Topological': 'Topological Features',
        'CTQW': 'CTQW Features',
        'Hybrid': 'Hybrid Features'
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    for i, (method_key, method_name) in enumerate(feature_methods.items()):
        try:
            # Extract features
            features, feature_names = extract_features_for_quantum(graphs)

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(graphs)-1))
            features_2d = tsne.fit_transform(features)

            # Plot t-SNE
            scatter = axes[i].scatter(features_2d[:, 0], features_2d[:, 1],
                                    c=labels, cmap='Set1', s=50, alpha=0.7)

            axes[i].set_xlabel('t-SNE 1')
            axes[i].set_ylabel('t-SNE 2')
            axes[i].set_title(f'{method_name}\nt-SNE Visualization', fontweight='bold')
            axes[i].grid(True, alpha=0.3)

            # Add colorbar
            plt.colorbar(scatter, ax=axes[i], label='Class Label')

        except Exception as e:
            logger.error(f"Failed to plot {method_key}: {e}")
            axes[i].text(0.5, 0.5, f'Failed to\ncompute\n{e}', ha='center', va='center',
                        transform=axes[i].transAxes, fontsize=10)
            axes[i].set_title(f'{method_name} (Failed)')

    # Feature importance heatmap (if available)
    try:
        # Compute feature importance using random forest
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        features, _ = extract_features_for_quantum(graphs)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Feature importance
        rf.fit(features, labels)
        importance = rf.feature_importances_

        # Plot feature importance
        ax = axes[3]
        feature_names_short = [f'F{i}' for i in range(len(importance))]
        ax.barh(range(len(importance)), importance, color='skyblue')
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(feature_names_short)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to compute feature importance: {e}")
        axes[3].text(0.5, 0.5, 'Feature\nImportance\nFailed', ha='center', va='center',
                    transform=axes[3].transAxes, fontsize=12)
        axes[3].set_title('Feature Importance (Failed)')

    # Method performance comparison
    try:
        ax = axes[4]

        # Get performance data for this dataset
        dataset_results = results[dataset_name]
        method_names = []
        accuracies = []

        for method_name, method_data in dataset_results.items():
            method_names.append(method_name.replace('_', '\n'))
            accuracies.append(method_data['accuracy_mean'])

        bars = ax.bar(method_names, accuracies, color='lightcoral', alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset_name} Method Comparison', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc".3f"}', ha='center', va='bottom', fontweight='bold')

        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create method comparison: {e}")
        axes[4].text(0.5, 0.5, 'Method\nComparison\nFailed', ha='center', va='center',
                    transform=axes[4].transAxes, fontsize=12)
        axes[4].set_title('Method Comparison (Failed)')

    # Quantum advantage analysis
    try:
        ax = axes[5]

        # Compare quantum vs classical performance
        quantum_methods = [m for m in dataset_results.keys() if 'quantum' in m.lower()]
        classical_methods = [m for m in dataset_results.keys() if 'quantum' not in m.lower()]

        if quantum_methods and classical_methods:
            quantum_scores = [dataset_results[m]['accuracy_mean'] for m in quantum_methods]
            classical_scores = [dataset_results[m]['accuracy_mean'] for m in classical_methods]

            # Create box plot comparison
            data = [classical_scores, quantum_scores]
            labels = ['Classical', 'Quantum']

            bp = ax.boxplot(data, labels=labels, patch_artist=True)

            # Color boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('Accuracy')
            ax.set_title('Classical vs Quantum Performance', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add individual points
            for i, scores in enumerate(data):
                y = scores
                x = np.random.normal(i + 1, 0.1, size=len(y))
                ax.scatter(x, y, alpha=0.8, s=30, color='black')

        else:
            ax.text(0.5, 0.5, 'No quantum\nmethods\navailable', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Classical vs Quantum (No Data)')

    except Exception as e:
        logger.error(f"Failed to create quantum advantage analysis: {e}")
        axes[5].text(0.5, 0.5, 'Quantum\nAdvantage\nFailed', ha='center', va='center',
                    transform=axes[5].transAxes, fontsize=12)
        axes[5].set_title('Quantum Advantage (Failed)')

    plt.tight_layout()
    plt.savefig(save_dir / f'{dataset_name}_feature_space_analysis.png', dpi=300, bbox_inches='tight')
    logger.info(f"Feature space analysis saved to {save_dir / f'{dataset_name}_feature_space_analysis.png'}")

    return fig


def plot_quantum_circuit_visualization(save_dir='comprehensive_results'):
    """
    Visualize quantum circuits used in the challenge.

    Args:
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    try:
        # Create simple visualization of quantum circuit architecture
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw quantum circuit schematic
        layers = ['Data\nEncoding', 'Entanglement', 'Variational\nLayers', 'Measurement']

        # Draw boxes for each layer
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f39c12']
        for i, (layer, color) in enumerate(zip(layers, colors)):
            # Main box
            rect = plt.Rectangle((i, 0), 0.8, 1, facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            # Layer label
            ax.text(i + 0.4, 0.5, layer, ha='center', va='center', fontweight='bold', fontsize=12)

            # Add specific operations
            if layer == 'Data\nEncoding':
                operations = ['RY(φ₁)', 'RY(φ₂)', '...', 'RY(φₙ)']
            elif layer == 'Entanglement':
                operations = ['CNOT', 'Ring', 'Star', 'Linear']
            elif layer == 'Variational\nLayers':
                operations = ['RY(θ)', 'RZ(θ)', 'CNOT', 'Repeat']
            else:
                operations = ['|ψ⟩', '⟨ψ|', 'K(i,j)']

            # Add operation details
            for j, op in enumerate(operations[:3]):  # Show first 3 operations
                ax.text(i + 0.4, 0.2 - j*0.15, op, ha='center', va='center', fontsize=9, alpha=0.8)

        # Draw quantum advantage highlight
        ax.arrow(2.5, 1.2, 0.8, 0, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=3)
        ax.text(3.0, 1.35, 'Quantum Advantage', ha='center', va='center', fontweight='bold', color='green', fontsize=14)

        # Set limits and labels
        ax.set_xlim(-0.5, len(layers) + 0.3)
        ax.set_ylim(-0.2, 1.5)
        ax.set_xlabel('Quantum Circuit Architecture')
        ax.set_title('Quantum Feature Map Architecture for QPoland Challenge', fontsize=16, fontweight='bold')
        ax.axis('off')

        # Add description
        description = """
        Quantum Feature Map Design:
        • Fidelity-based kernels: K(i,j) = |⟨ψᵢ|ψⱼ⟩|²
        • Quantum walk embeddings inspired by CTQW
        • Parameterized circuits for molecular graphs
        • Hybrid classical-quantum feature spaces
        """
        ax.text(-0.3, -0.1, description, fontsize=10, style='italic', wrap=True)

        plt.tight_layout()
        plt.savefig(save_dir / 'quantum_circuit_architecture.png', dpi=300, bbox_inches='tight')
        logger.info(f"Quantum circuit visualization saved to {save_dir / 'quantum_circuit_architecture.png'}")

        return fig

    except Exception as e:
        logger.error(f"Failed to create quantum circuit visualization: {e}")
        return None


def plot_comprehensive_benchmark_summary(results, save_dir='comprehensive_results'):
    """
    Create comprehensive summary plots for the entire benchmark.

    Args:
        results: Complete results dictionary
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    logger.info("Creating comprehensive benchmark visualizations...")

    # 1. Quantum vs Classical comparison
    fig1 = plot_quantum_vs_classical_comparison(results, save_dir)

    # 2. Individual dataset analysis (for first available dataset)
    available_datasets = list(results.keys())
    if available_datasets:
        fig2 = plot_kernel_matrices_comparison(results, available_datasets[0], save_dir)
        fig3 = plot_feature_space_analysis(results, available_datasets[0], save_dir)

    # 3. Quantum circuit architecture
    fig4 = plot_quantum_circuit_visualization(save_dir)

    # 4. Summary table as plot
    fig5, ax = plt.subplots(figsize=(12, 8))

    # Create performance summary table
    summary_data = []
    for dataset_name, dataset_results in results.items():
        for method_name, method_results in dataset_results.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Method': method_name,
                'Accuracy': method_results['accuracy_mean'],
                'F1': method_results['f1_mean'],
                'Type': method_results.get('method_type', 'unknown')
            })

    df = pd.DataFrame(summary_data)

    # Create pivot table for heatmap
    pivot_acc = df.pivot_table(values='Accuracy', index='Method', columns='Dataset', aggfunc='mean')
    pivot_f1 = df.pivot_table(values='F1', index='Method', columns='Dataset', aggfunc='mean')

    # Plot accuracy heatmap
    sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Accuracy'})

    ax.set_title('Comprehensive Performance Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Method')

    plt.tight_layout()
    plt.savefig(save_dir / 'comprehensive_performance_heatmap.png', dpi=300, bbox_inches='tight')
    logger.info(f"Performance heatmap saved to {save_dir / 'comprehensive_performance_heatmap.png'}")

    logger.info("Comprehensive visualizations completed!")
    return [fig1, fig2, fig3, fig4, fig5] if available_datasets else [fig1, fig4, fig5]


if __name__ == "__main__":
    # Test visualization functions
    logging.basicConfig(level=logging.INFO)

    # Create sample results for testing
    sample_results = {
        'MUTAG': {
            'Topological_RBF': {
                'accuracy_mean': 0.85, 'accuracy_std': 0.05,
                'f1_mean': 0.83, 'f1_std': 0.06,
                'method_type': 'classical', 'best_C': 1.0, 'n_features': 13
            },
            'Quantum_Fidelity': {
                'accuracy_mean': 0.89, 'accuracy_std': 0.04,
                'f1_mean': 0.88, 'f1_std': 0.05,
                'method_type': 'quantum', 'best_C': 1.0, 'n_features': 25
            }
        }
    }

    logger.info("Testing enhanced visualizations...")
    try:
        # Test quantum vs classical comparison
        fig1 = plot_quantum_vs_classical_comparison(sample_results)
        plt.close(fig1)

        # Test kernel matrices
        fig2 = plot_kernel_matrices_comparison(sample_results, 'MUTAG')
        plt.close(fig2)

        # Test feature space analysis
        fig3 = plot_feature_space_analysis(sample_results, 'MUTAG')
        plt.close(fig3)

        # Test quantum circuit visualization
        fig4 = plot_quantum_circuit_visualization()
        plt.close(fig4)

        logger.info("All visualization functions working correctly!")

    except Exception as e:
        logger.error(f"Visualization test failed: {e}")
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            projections[mask, 0],
            projections[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )
    
    ax.set_xlabel(f'PC1 (λ={eigenvals[0]:.3f})', fontsize=12)
    ax.set_ylabel(f'PC2 (λ={eigenvals[1]:.3f})', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Kernel PCA plot saved to {save_path}")
    
    return fig


def plot_tsne_embedding(features, labels, title='t-SNE Embedding', 
                       save_path=None, figsize=(10, 8), perplexity=30):
    """
    Plot t-SNE embedding of features.
    
    Args:
        features: Feature matrix (n x d)
        labels: Class labels
        title: Plot title
        save_path: Where to save figure
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
    """
    logger.info("Computing t-SNE embedding...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(features)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"t-SNE plot saved to {save_path}")
    
    return fig


def plot_cv_results(results_dict, metric='accuracy', save_path=None, figsize=(12, 6)):
    """
    Plot cross-validation results comparison.
    
    Args:
        results_dict: Dictionary of {name: results} pairs
        metric: Which metric to plot ('accuracy' or 'f1')
        save_path: Where to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(results_dict.keys())
    means = [results_dict[name][f'{metric}_mean'] for name in names]
    stds = [results_dict[name][f'{metric}_std'] for name in names]
    
    x = np.arange(len(names))
    
    # Bar plot with error bars
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                  color=plt.cm.Set3(np.linspace(0, 1, len(names))))
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'{metric.capitalize()} Comparison (10-Fold CV)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"CV results plot saved to {save_path}")
    
    return fig


def plot_feature_importance(importance_df, top_k=15, save_path=None, figsize=(10, 8)):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_k: Number of top features to show
        save_path: Where to save figure
        figsize: Figure size
    """
    top_features = importance_df.head(top_k)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_k} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                         save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Where to save figure
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def create_results_dashboard(results_dict, features, labels, output_dir='results/viz'):
    """
    Create a comprehensive dashboard of visualizations.
    
    Args:
        results_dict: Dictionary of {name: results} pairs
        features: Feature matrix
        labels: Class labels
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating visualization dashboard in {output_dir}...")
    
    # 1. CV results comparison
    plot_cv_results(results_dict, metric='accuracy',
                   save_path=output_path / 'cv_accuracy_comparison.png')
    
    plot_cv_results(results_dict, metric='f1',
                   save_path=output_path / 'cv_f1_comparison.png')
    
    # 2. t-SNE embedding
    plot_tsne_embedding(features, labels, title='Feature Space (t-SNE)',
                       save_path=output_path / 'tsne_embedding.png')
    
    # 3. Feature importance (if available)
    from qkernels.utils import compute_feature_importance
    try:
        importance_df = compute_feature_importance(
            features, labels, 
            feature_names=[f'Feature {i}' for i in range(features.shape[1])]
        )
        plot_feature_importance(importance_df,
                               save_path=output_path / 'feature_importance.png')
    except:
        logger.warning("Could not compute feature importance")
    
    logger.info(f"Dashboard created successfully in {output_dir}")


if __name__ == "__main__":
    # Test visualizations
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 3, n_samples)
    
    # Create kernel matrix
    from sklearn.metrics.pairwise import rbf_kernel
    K = rbf_kernel(features)
    
    # Test plots
    plot_kernel_matrix(K, labels, title='Test Kernel Matrix')
    plot_kernel_pca(K, labels, title='Test Kernel PCA')
    plot_tsne_embedding(features, labels, title='Test t-SNE')
    
    plt.show()
    
    print("Visualization tests completed!")
