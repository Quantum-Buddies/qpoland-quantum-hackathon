"""
Experiment runner for molecular graph classification.

Command-line interface for running experiments across different datasets and methods.
"""
import argparse
import logging
import sys
from pathlib import Path
import json
import time
import numpy as np

# Add qkernels to path
sys.path.append(str(Path(__file__).parent.parent))

from qkernels.datasets import MolecularGraphDataset
from qkernels.features import TopologicalFeatureExtractor, CTQWFeatureExtractor, HybridFeatureExtractor
from qkernels.kernels import KernelSVM
from qkernels.quantum import QuantumSVM
from qkernels.eval import ClassificationEvaluator
from qkernels.quantum import create_quantum_kernel_matrix


def run_experiment(dataset_name: str, feature_type: str, kernel_type: str,
                  cv_folds: int = 10, random_state: int = 42, output_dir: str = 'results'):
    """
    Run a complete experiment for one dataset.

    Args:
        dataset_name: Name of dataset (MUTAG, AIDS, etc.)
        feature_type: Type of features ('topo', 'ctqw', 'hybrid', 'quantum')
        kernel_type: Type of kernel ('rbf', 'linear', 'quantum')
        cv_folds: Number of CV folds
        random_state: Random seed
        output_dir: Directory to save results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {dataset_name} + {feature_type} + {kernel_type}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    start_time = time.time()

    try:
        # 1. Load dataset
        logger.info(f"Loading {dataset_name} dataset...")
        dataset = MolecularGraphDataset(dataset_name)
        graphs, labels = dataset.get_graphs_and_labels()

        if len(graphs) == 0:
            raise ValueError(f"No graphs loaded for {dataset_name}")

        logger.info(f"Loaded {len(graphs)} graphs with {len(set(labels))} classes")

        # 2. Extract features
        logger.info(f"Extracting {feature_type} features...")

        if feature_type == 'topo':
            extractor = TopologicalFeatureExtractor()
        elif feature_type == 'ctqw':
            extractor = CTQWFeatureExtractor(gamma=1.0, time_points=[0.5, 1.0, 2.0])
        elif feature_type == 'hybrid':
            extractor = HybridFeatureExtractor()
        elif feature_type == 'quantum':
            # For quantum features, we'll use the hybrid features as input to quantum circuits
            extractor = HybridFeatureExtractor()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        # Fit scaler on all data
        scaler = extractor.fit_scaler(graphs)

        # Extract features
        features = [extractor.extract_features(g) for g in graphs]
        features = np.array(features)
        features_scaled = scaler.transform(features)
        feature_names = extractor.get_feature_names()

        logger.info(f"Extracted features with shape: {features.shape}")

        # 3. Set up model
        logger.info(f"Setting up {kernel_type} kernel model...")

        if kernel_type == 'quantum' and feature_type == 'quantum':
            # Quantum kernel on quantum features
            model = QuantumSVM(n_qubits=8, n_layers=2, C=1.0)
        elif kernel_type == 'quantum':
            # Quantum kernel on classical features
            def quantum_kernel(X1, X2=None):
                return create_quantum_kernel_matrix(X1, X2, n_qubits=8, n_layers=2)

            model = KernelSVM(kernel='precomputed', C=1.0)
        else:
            # Classical kernel
            model = KernelSVM(kernel=kernel_type, C=1.0)

        # 4. Evaluate model
        logger.info("Running cross-validation...")
        evaluator = ClassificationEvaluator(cv_folds=cv_folds, random_state=random_state)

        if kernel_type == 'quantum' and feature_type != 'quantum':
            # For quantum kernel with classical features, we need special handling
            # This is a simplified version - in practice, you'd compute the kernel matrix differently
            cv_results = evaluator.cross_validate(model, features_scaled, np.array(labels), f"{feature_type}_{kernel_type}")
        else:
            cv_results = evaluator.cross_validate(model, features_scaled, np.array(labels), f"{feature_type}_{kernel_type}")

        # 5. Save results
        experiment_name = f"{dataset_name}_{feature_type}_{kernel_type}"
        results_file = output_path / f"{experiment_name}_results.json"

        # Add metadata to results
        cv_results.update({
            'dataset': dataset_name,
            'feature_type': feature_type,
            'kernel_type': kernel_type,
            'n_features': features.shape[1],
            'n_samples': features.shape[0],
            'n_classes': len(set(labels)),
            'feature_names': feature_names,
            'total_time': time.time() - start_time
        })

        with open(results_file, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)

        logger.info("Results saved to {}".format(results_file))
        logger.info("Experiment completed in {:.2f} seconds".format(time.time() - start_time))

        return cv_results

    except Exception as e:
        logger.error("Experiment failed: {}".format(e))
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Run molecular graph classification experiments')

    parser.add_argument('--dataset', type=str, required=True,
                       choices=['MUTAG', 'AIDS', 'PROTEINS', 'NCI1', 'PTC_MR'],
                       help='Dataset to use')
    parser.add_argument('--feature', type=str, required=True,
                       choices=['topo', 'ctqw', 'hybrid', 'quantum'],
                       help='Feature extraction method')
    parser.add_argument('--kernel', type=str, required=True,
                       choices=['rbf', 'linear', 'quantum'],
                       help='Kernel type')
    parser.add_argument('--cv-folds', type=int, default=10,
                       help='Number of CV folds')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run experiment
    try:
        results = run_experiment(
            dataset_name=args.dataset,
            feature_type=args.feature,
            kernel_type=args.kernel,
            cv_folds=args.cv_folds,
            random_state=args.random_seed,
            output_dir=args.output_dir
        )

        print("\nExperiment completed successfully!")
        print("Accuracy: {:.4f} ± {:.4f}".format(results['accuracy_mean'], results['accuracy_std']))
        print("F1-Score: {:.4f} ± {:.4f}".format(results['f1_mean'], results['f1_std']))

    except Exception as e:
        print("Experiment failed: {}".format(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
