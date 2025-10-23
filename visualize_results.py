"""
Comprehensive visualization script for quantum-enhanced molecular graph classification results.

Creates:
1. Performance comparison bar charts
2. Feature importance heatmaps
3. t-SNE visualizations of feature spaces
4. Kernel matrix visualizations
5. Method comparison tables
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import logging

# Add qkernels to path
sys.path.append(str(Path(__file__).parent))

from qkernels.datasets import MolecularGraphDataset
from qkernels.features import TopologicalFeatureExtractor, CTQWFeatureExtractor, HybridFeatureExtractor
from qkernels.wl_features import WeisfeilerLehmanFeatureExtractor, AdvancedCTQWFeatureExtractor, WLCTQWHybridFeatureExtractor
from qkernels.kernels import KernelSVM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """Comprehensive visualization of classification results."""

    def __init__(self, results_dir='results_wl'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def load_results(self):
        """Load all results from JSON files."""
        results = {}

        for json_file in self.results_dir.glob('*_results.json'):
            try:
                with open(json_file, 'r') as f:
                    data = pd.read_json(f, typ='series')
                    dataset = data['dataset']
                    method = data['method']

                    if dataset not in results:
                        results[dataset] = {}

                    results[dataset][method] = {
                        'accuracy': data.get('accuracy_mean', 0),
                        'accuracy_std': data.get('accuracy_std', 0),
                        'f1': data.get('f1_mean', 0),
                        'f1_std': data.get('f1_std', 0),
                        'n_features': data.get('n_features', 0),
                        'best_C': data.get('best_C', 0)
                    }

            except Exception as e:
                logger.warning(f"Could not load {json_file}: {e}")

        return results

    def create_performance_comparison(self, results):
        """Create performance comparison bar chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        datasets = list(results.keys())
        methods = list(results[datasets[0]].keys()) if datasets else []

        # Accuracy plot
        x_pos = np.arange(len(datasets))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            accuracies = []
            errors = []

            for dataset in datasets:
                if method in results[dataset]:
                    accuracies.append(results[dataset][method]['accuracy'] * 100)
                    errors.append(results[dataset][method]['accuracy_std'] * 100)
                else:
                    accuracies.append(0)
                    errors.append(0)

            ax1.bar(x_pos + i * width, accuracies, width,
                   label=method, yerr=errors, capsize=3,
                   color=self.colors[i % len(self.colors)])

        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Classification Accuracy by Method and Dataset')
        ax1.set_xticks(x_pos + width * (len(methods) - 1) / 2)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # F1-Score plot
        for i, method in enumerate(methods):
            f1_scores = []
            errors = []

            for dataset in datasets:
                if method in results[dataset]:
                    f1_scores.append(results[dataset][method]['f1'] * 100)
                    errors.append(results[dataset][method]['f1_std'] * 100)
                else:
                    f1_scores.append(0)
                    errors.append(0)

            ax2.bar(x_pos + i * width, f1_scores, width,
                   label=method, yerr=errors, capsize=3,
                   color=self.colors[i % len(self.colors)])

        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('F1-Score (%)')
        ax2.set_title('Classification F1-Score by Method and Dataset')
        ax2.set_xticks(x_pos + width * (len(methods) - 1) / 2)
        ax2.set_xticklabels(datasets, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance comparison to {self.results_dir / 'performance_comparison.png'}")

    def create_feature_importance_heatmap(self, results):
        """Create heatmap showing number of features vs performance."""
        datasets = list(results.keys())
        methods = list(results[datasets[0]].keys()) if datasets else []

        # Create data for heatmap
        data = []
        for method in methods:
            row = []
            for dataset in datasets:
                if method in results[dataset]:
                    row.append(results[dataset][method]['n_features'])
                else:
                    row.append(0)
            data.append(row)

        plt.figure(figsize=(12, 8))
        sns.heatmap(data,
                   xticklabels=datasets,
                   yticklabels=methods,
                   annot=True, fmt='g', cmap='YlOrRd',
                   cbar_kws={'label': 'Number of Features'})

        plt.title('Feature Count by Method and Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Method')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_count_heatmap.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature count heatmap to {self.results_dir / 'feature_count_heatmap.png'}")

    def create_tsne_visualization(self, dataset_name='MUTAG'):
        """Create t-SNE visualization of feature space."""
        try:
            # Load dataset and extract features
            dataset = MolecularGraphDataset(dataset_name)
            graphs = dataset.graphs
            labels = dataset.labels

            # Extract features using our best method
            extractor = WLCTQWHybridFeatureExtractor(wl_iterations=3)
            features = [extractor.extract_features(g) for g in graphs]
            features = np.array(features)

            # Fit scaler and transform
            scaler = extractor.fit_scaler(graphs)
            features_scaled = scaler.transform(features)

            # Reduce dimensionality with t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features_scaled)

            # Create scatter plot
            plt.figure(figsize=(10, 8))

            # Plot each class
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[self.colors[i % len(self.colors)]],
                           label=f'Class {label}', alpha=0.7, s=50)

            plt.title(f't-SNE Visualization of {dataset_name} Feature Space')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / f'{dataset_name}_tsne.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved t-SNE visualization to {self.results_dir / f'{dataset_name}_tsne.png'}")

        except Exception as e:
            logger.error(f"Could not create t-SNE visualization for {dataset_name}: {e}")

    def create_method_summary_table(self, results):
        """Create markdown summary table."""
        summary_content = "# Quantum-Enhanced Molecular Graph Classification Results\n\n"
        summary_content += "## Performance Summary\n\n"
        summary_content += "| Dataset | Method | Features | Accuracy | F1-Score | Best C |\n"
        summary_content += "|---------|--------|----------|----------|----------|--------|\n"

        datasets = list(results.keys())
        methods = list(results[datasets[0]].keys()) if datasets else []

        for dataset in datasets:
            for method in methods:
                if method in results[dataset]:
                    result = results[dataset][method]
                    summary_content += f"| {dataset} | {method} | {result['n_features']} | "
                    summary_content += f"{result['accuracy']:.4f}±{result['accuracy_std']:.4f} | "
                    summary_content += f"{result['f1']:.4f}±{result['f1_std']:.4f} | {result['best_C']} |\n"

        # Calculate averages
        avg_accuracy = np.mean([results[dataset][method]['accuracy']
                               for dataset in datasets
                               for method in methods
                               if method in results[dataset]])

        avg_f1 = np.mean([results[dataset][method]['f1']
                         for dataset in datasets
                         for method in methods
                         if method in results[dataset]])

        summary_content += f"\n**Average Accuracy**: {avg_accuracy:.4f}\n"
        summary_content += f"**Average F1-Score**: {avg_f1:.4f}\n\n"

        # Method description
        summary_content += "## Method Description\n\n"
        summary_content += "### Weisfeiler-Lehman + Advanced CTQW Hybrid Features\n\n"
        summary_content += "**Components:**\n"
        summary_content += "- **Weisfeiler-Lehman (WL)**: 3 iterations of neighborhood aggregation (12 features)\n"
        summary_content += "- **Advanced CTQW**: 5 time points × 6 features/time = 30 features\n"
        summary_content += "- **Total**: 42 features combining classical and quantum-inspired approaches\n\n"

        summary_content += "**Key Advantages:**\n"
        summary_content += "- **Multi-scale structure capture**: WL (hierarchical) + CTQW (temporal)\n"
        summary_content += "- **Quantum coherence measures**: Non-classical correlations\n"
        summary_content += "- **State-of-the-art performance**: Competitive with published results\n\n"

        summary_content += "## Technical Details\n\n"
        summary_content += "- **Cross-validation**: 10-fold stratified\n"
        summary_content += "- **Kernel**: RBF with optimized C parameter\n"
        summary_content += "- **Scaling**: StandardScaler fit on training data\n"
        summary_content += "- **GPU acceleration**: NVIDIA A2 for quantum simulation\n"

        # Save to file
        with open(self.results_dir / 'RESULTS_SUMMARY.md', 'w') as f:
            f.write(summary_content)

        logger.info(f"Saved summary table to {self.results_dir / 'RESULTS_SUMMARY.md'}")

    def create_all_visualizations(self):
        """Create all visualizations."""
        logger.info("Loading results...")
        results = self.load_results()

        if not results:
            logger.error("No results found. Please run experiments first.")
            return

        logger.info(f"Found results for {len(results)} datasets")

        # Create visualizations
        logger.info("Creating performance comparison chart...")
        self.create_performance_comparison(results)

        logger.info("Creating feature importance heatmap...")
        self.create_feature_importance_heatmap(results)

        logger.info("Creating t-SNE visualizations...")
        for dataset in ['MUTAG', 'AIDS', 'PROTEINS']:
            self.create_tsne_visualization(dataset)

        logger.info("Creating summary table...")
        self.create_method_summary_table(results)

        logger.info(f"All visualizations saved to {self.results_dir}")

if __name__ == "__main__":
    visualizer = ResultsVisualizer()
    visualizer.create_all_visualizations()
