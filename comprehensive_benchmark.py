"""
Comprehensive benchmark script for QPoland Quantum Hackathon Challenge.

Compares:
1. Classical baselines (Weisfeiler-Lehman, Shortest-Path, Graphlet kernels)
2. Topological features with RBF kernel
3. CTQW features with RBF kernel
4. Hybrid features with RBF kernel
5. Quantum fidelity kernels
6. Quantum walk embeddings

All evaluated on the 5 required datasets: MUTAG, AIDS, PROTEINS, NCI1, PTC_MR
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add qkernels to path
sys.path.append(str(Path(__file__).parent))

from qkernels.datasets import MolecularGraphDataset
from qkernels.features import TopologicalFeatureExtractor, CTQWFeatureExtractor, HybridFeatureExtractor
from qkernels.wl_features import WeisfeilerLehmanFeatureExtractor, AdvancedCTQWFeatureExtractor, WLCTQWHybridFeatureExtractor
from qkernels.classical_baselines import create_classical_kernel_matrix, WeisfeilerLehmanKernel, ShortestPathKernel, GraphletKernel
from qkernels.quantum import create_fidelity_kernel_matrix, create_quantum_walk_kernel_matrix, QuantumSVM
from qkernels.kernels import KernelSVM
from qkernels.eval import ClassificationEvaluator
from qkernels.utils import setup_logging, print_results_table

# Set up logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveBenchmark:
    """Complete benchmark comparing all methods for the QPoland challenge."""

    def __init__(self, data_dir='data', results_dir='comprehensive_results'):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # All required datasets
        self.datasets = ['MUTAG', 'AIDS', 'PROTEINS', 'NCI1', 'PTC_MR']

        # All methods to compare
        self.methods = {
            'WL_Classical': {'type': 'classical', 'kernel': 'wl', 'features': None},
            'SP_Classical': {'type': 'classical', 'kernel': 'sp', 'features': None},
            'Graphlet_Classical': {'type': 'classical', 'kernel': 'graphlet', 'features': None},
            'Topological_RBF': {'type': 'classical', 'kernel': 'rbf', 'features': 'topological'},
            'CTQW_RBF': {'type': 'classical', 'kernel': 'rbf', 'features': 'ctqw'},
            'Hybrid_RBF': {'type': 'classical', 'kernel': 'rbf', 'features': 'hybrid'},
            'WL_CTQW_Hybrid': {'type': 'classical', 'kernel': 'rbf', 'features': 'wl_ctqw_hybrid'},
            'Quantum_Fidelity': {'type': 'quantum', 'kernel': 'fidelity', 'features': None},
            'Quantum_Walk': {'type': 'quantum', 'kernel': 'quantum_walk', 'features': None},
        }

        # Cross-validation parameters
        self.cv_folds = 10
        self.random_state = 42
        self.C_values = [0.1, 1.0, 10.0]

    def load_datasets(self):
        """Load all required datasets."""
        logger.info("Loading datasets...")
        datasets = {}

        for name in self.datasets:
            try:
                logger.info(f"Loading {name}...")
                dataset = MolecularGraphDataset(name, self.data_dir)
                graphs, labels = dataset.get_graphs_and_labels()

                datasets[name] = {
                    'graphs': graphs,
                    'labels': np.array(labels),
                    'n_graphs': len(graphs),
                    'n_classes': len(set(labels))
                }

                logger.info(f"✅ {name}: {len(graphs)} graphs, {len(set(labels))} classes")

            except Exception as e:
                logger.error(f"❌ Failed to load {name}: {e}")
                continue

        return datasets

    def extract_features(self, graphs, method):
        """Extract features based on method type."""
        if method == 'topological':
            extractor = TopologicalFeatureExtractor()
        elif method == 'ctqw':
            extractor = CTQWFeatureExtractor(gamma=1.0, time_points=[0.5, 1.0, 2.0])
        elif method == 'hybrid':
            extractor = HybridFeatureExtractor()
        elif method == 'wl_ctqw_hybrid':
            extractor = WLCTQWHybridFeatureExtractor()
        else:
            raise ValueError(f"Unknown feature method: {method}")

        # Fit and transform
        scaler = extractor.fit_scaler(graphs)
        features = np.array([extractor.extract_features(g) for g in graphs])
        features_scaled = scaler.transform(features)

        return features_scaled, extractor.get_feature_names()

    def run_classical_baseline(self, graphs, labels, method_name, params):
        """Run classical baseline methods."""
        logger.info(f"Running {method_name}...")

        if params['kernel'] == 'wl':
            kernel = WeisfeilerLehmanKernel(h=3)
        elif params['kernel'] == 'sp':
            kernel = ShortestPathKernel()
        elif params['kernel'] == 'graphlet':
            kernel = GraphletKernel(k=3)
        else:
            raise ValueError(f"Unknown classical kernel: {params['kernel']}")

        # Hyperparameter tuning
        best_score = 0
        best_C = 1.0
        best_results = None

        for C in self.C_values:
            # Compute kernel matrix
            K_train = kernel.fit(graphs).transform(graphs)

            # SVM with precomputed kernel
            from sklearn.svm import SVC
            svm = SVC(kernel='precomputed', C=C, random_state=self.random_state)

            # Cross-validation
            evaluator = ClassificationEvaluator(cv_folds=self.cv_folds, random_state=self.random_state)
            results = evaluator.cross_validate(svm, K_train, labels, model_name=f"{method_name}_C{C}")

            if results['accuracy_mean'] > best_score:
                best_score = results['accuracy_mean']
                best_C = C
                best_results = results

        return best_results

    def run_feature_based_method(self, graphs, labels, method_name, params):
        """Run feature-based methods with RBF kernel."""
        logger.info(f"Running {method_name}...")

        # Extract features
        features, feature_names = self.extract_features(graphs, params['features'])
        n_features = features.shape[1]

        # Hyperparameter tuning
        best_score = 0
        best_C = 1.0
        best_results = None

        for C in self.C_values:
            # RBF kernel SVM
            svm = KernelSVM(kernel='rbf', C=C, gamma='scale')

            # Cross-validation
            evaluator = ClassificationEvaluator(cv_folds=self.cv_folds, random_state=self.random_state)
            results = evaluator.cross_validate(
                svm, features, labels,
                model_name=f"{method_name}_C{C}"
            )

            if results['accuracy_mean'] > best_score:
                best_score = results['accuracy_mean']
                best_C = C
                best_results = results

        # Add feature information to results
        best_results['n_features'] = n_features
        best_results['feature_names'] = feature_names

        return best_results

    def run_quantum_method(self, graphs, labels, method_name, params):
        """Run quantum methods."""
        logger.info(f"Running {method_name}...")

        # Quantum methods work directly with graphs or features
        if params['kernel'] == 'fidelity':
            # Extract features first for quantum fidelity kernel
            features, feature_names = self.extract_features(graphs, 'hybrid')
            n_features = features.shape[1]

            # Quantum fidelity kernel
            best_score = 0
            best_C = 1.0
            best_results = None

            for C in self.C_values:
                try:
                    qsvm = QuantumSVM(n_qubits=4, n_layers=1, C=C, kernel_type='fidelity')

                    # Cross-validation with quantum kernel
                    evaluator = ClassificationEvaluator(cv_folds=self.cv_folds, random_state=self.random_state)

                    # For quantum methods, use a simplified CV approach
                    results = evaluator.cross_validate(
                        qsvm, features, labels,
                        model_name=f"{method_name}_C{C}"
                    )

                    if results['accuracy_mean'] > best_score:
                        best_score = results['accuracy_mean']
                        best_C = C
                        best_results = results

                except Exception as e:
                    logger.warning(f"Quantum method {method_name} failed: {e}")
                    continue

            if best_results:
                best_results['n_features'] = n_features
                return best_results

        elif params['kernel'] == 'quantum_walk':
            # Quantum walk kernel
            best_score = 0
            best_C = 1.0
            best_results = None

            for C in self.C_values:
                try:
                    qsvm = QuantumSVM(n_qubits=4, n_layers=1, C=C, kernel_type='quantum_walk')

                    # Cross-validation
                    evaluator = ClassificationEvaluator(cv_folds=self.cv_folds, random_state=self.random_state)
                    results = evaluator.cross_validate(
                        qsvm, graphs, labels,
                        model_name=f"{method_name}_C{C}"
                    )

                    if results['accuracy_mean'] > best_score:
                        best_score = results['accuracy_mean']
                        best_C = C
                        best_results = results

                except Exception as e:
                    logger.warning(f"Quantum walk method {method_name} failed: {e}")
                    continue

            return best_results

        # Fallback to classical method if quantum fails
        logger.warning(f"Quantum method {method_name} failed, using classical fallback...")
        return self.run_feature_based_method(graphs, labels, f"{method_name}_Classical", params)

    def run_benchmark(self):
        """Run complete benchmark on all datasets and methods."""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE QPOLAND QUANTUM HACKATHON BENCHMARK")
        logger.info("=" * 80)

        # Load datasets
        datasets = self.load_datasets()

        # Results storage
        all_results = {}

        for dataset_name, dataset_info in datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"BENCHMARKING DATASET: {dataset_name}")
            logger.info(f"{'='*60}")

            graphs = dataset_info['graphs']
            labels = dataset_info['labels']
            n_graphs = dataset_info['n_graphs']
            n_classes = dataset_info['n_classes']

            logger.info(f"Dataset: {n_graphs} graphs, {n_classes} classes")
            logger.info(f"Class distribution: {np.bincount(labels)}")

            dataset_results = {}

            # Run all methods
            for method_name, params in self.methods.items():
                try:
                    if params['type'] == 'classical' and params['features'] is None:
                        # Classical kernel methods
                        results = self.run_classical_baseline(graphs, labels, method_name, params)
                    elif params['type'] == 'classical' and params['features'] is not None:
                        # Feature-based classical methods
                        results = self.run_feature_based_method(graphs, labels, method_name, params)
                    elif params['type'] == 'quantum':
                        # Quantum methods
                        results = self.run_quantum_method(graphs, labels, method_name, params)
                    else:
                        logger.warning(f"Unknown method type for {method_name}")
                        continue

                    if results:
                        # Add metadata
                        results['dataset'] = dataset_name
                        results['method'] = method_name
                        results['method_type'] = params['type']
                        results['n_graphs'] = n_graphs
                        results['n_classes'] = n_classes

                        dataset_results[method_name] = results

                        logger.info(f"Completed {method_name}: Acc={results['accuracy_mean']:.4f}±{results['accuracy_std']:.4f}, "
                                  f"F1={results['f1_mean']:.4f}±{results['f1_std']:.4f}")

                except Exception as e:
                    logger.error(f"Method {method_name} failed: {e}")
                    continue

            all_results[dataset_name] = dataset_results

        # Save results
        self.save_results(all_results)

        # Generate summary report
        self.generate_summary_report(all_results)

        return all_results

    def save_results(self, results):
        """Save detailed results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for dataset_name, dataset_results in results.items():
            # Save individual dataset results
            result_file = self.results_dir / f"{dataset_name}_comprehensive_{timestamp}.json"

            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for method_name, method_results in dataset_results.items():
                serializable_results[method_name] = {}
                for key, value in method_results.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[method_name][key] = value.tolist()
                    elif isinstance(value, (np.int64, np.int32)):
                        serializable_results[method_name][key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        serializable_results[method_name][key] = float(value)
                    else:
                        serializable_results[method_name][key] = value

            with open(result_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)

        # Save comprehensive summary
        summary_file = self.results_dir / f"comprehensive_benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {self.results_dir}")

    def generate_summary_report(self, results):
        """Generate comprehensive summary report."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE BENCHMARK SUMMARY")
        logger.info("=" * 80)

        # Create summary table
        summary_data = []

        for dataset_name, dataset_results in results.items():
            for method_name, method_results in dataset_results.items():
                summary_data.append({
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Type': method_results.get('method_type', 'unknown'),
                    'Accuracy': f"{method_results['accuracy_mean']:.4f}±{method_results['accuracy_std']:.4f}",
                    'F1-Score': f"{method_results['f1_mean']:.4f}±{method_results['f1_std']:.4f}",
                    'Best_C': method_results.get('best_C', 1.0),
                    'Features': method_results.get('n_features', 'N/A')
                })

        df = pd.DataFrame(summary_data)

        # Save to CSV
        csv_file = self.results_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Summary saved to {csv_file}")

        # Print best methods per dataset
        logger.info("\nBEST METHODS PER DATASET:")
        for dataset_name in self.datasets:
            if dataset_name in results:
                best_method = max(
                    results[dataset_name].items(),
                    key=lambda x: x[1]['accuracy_mean']
                )
                logger.info(f"{dataset_name}: {best_method[0]} (Acc: {best_method[1]['accuracy_mean']:.4f})")

        # Print overall best method
        all_scores = []
        for dataset_results in results.values():
            for method_results in dataset_results.values():
                all_scores.append(method_results['accuracy_mean'])

        if all_scores:
            overall_avg = np.mean(all_scores)
            logger.info(f"\nOVERALL AVERAGE ACCURACY: {overall_avg:.4f}")

        # Print quantum vs classical comparison
        quantum_scores = []
        classical_scores = []

        for dataset_results in results.values():
            for method_name, method_results in dataset_results.items():
                if method_results.get('method_type') == 'quantum':
                    quantum_scores.append(method_results['accuracy_mean'])
                else:
                    classical_scores.append(method_results['accuracy_mean'])

        if quantum_scores and classical_scores:
            quantum_avg = np.mean(quantum_scores)
            classical_avg = np.mean(classical_scores)

            logger.info(f"\nQUANTUM METHODS AVERAGE: {quantum_avg:.4f}")
            logger.info(f"CLASSICAL METHODS AVERAGE: {classical_avg:.4f}")
            if quantum_avg > classical_avg:
                logger.info(f"QUANTUM ADVANTAGE: +{quantum_avg - classical_avg:.4f}")
            else:
                logger.info(f"CLASSICAL PERFORMANCE: +{classical_avg - quantum_avg:.4f}")

        return df


def main():
    """Run comprehensive benchmark."""
    # Initialize benchmark
    benchmark = ComprehensiveBenchmark()

    # Run benchmark
    results = benchmark.run_benchmark()

    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE BENCHMARK COMPLETED")
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
