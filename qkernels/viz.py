"""
Visualization utilities for quantum molecular graph classification.

Includes kernel matrix heatmaps, t-SNE embeddings, and performance plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_kernel_matrix(K, labels=None, title='Kernel Matrix', save_path=None, figsize=(10, 8)):
    """
    Plot kernel matrix as heatmap.
    
    Args:
        K: Kernel matrix (n x n)
        labels: Optional class labels for sorting
        title: Plot title
        save_path: Where to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by labels if provided
    if labels is not None:
        sort_idx = np.argsort(labels)
        K_sorted = K[sort_idx][:, sort_idx]
        
        # Add class boundaries
        unique_labels = np.unique(labels)
        boundaries = []
        for label in unique_labels[:-1]:
            boundary = np.sum(labels[sort_idx] == label)
            boundaries.append(boundary)
        
        # Cumulative boundaries
        boundaries = np.cumsum(boundaries)
    else:
        K_sorted = K
        boundaries = []
    
    # Plot heatmap
    im = ax.imshow(K_sorted, cmap='viridis', aspect='auto')
    
    # Add boundaries
    for boundary in boundaries:
        ax.axhline(y=boundary - 0.5, color='red', linewidth=2, linestyle='--')
        ax.axvline(x=boundary - 0.5, color='red', linewidth=2, linestyle='--')
    
    # Colorbar
    plt.colorbar(im, ax=ax, label='Kernel Value')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Kernel matrix plot saved to {save_path}")
    
    return fig


def plot_kernel_pca(K, labels, title='Kernel PCA', save_path=None, figsize=(10, 8)):
    """
    Plot kernel PCA visualization.
    
    Args:
        K: Kernel matrix (n x n)
        labels: Class labels
        title: Plot title
        save_path: Where to save figure
        figsize: Figure size
    """
    # Center kernel matrix
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(K_centered)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Project onto first 2 principal components
    projections = eigenvecs[:, :2]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
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
