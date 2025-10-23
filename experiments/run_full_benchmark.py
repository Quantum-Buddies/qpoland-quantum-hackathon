"""
Complete benchmark script for QPoland Hackathon Challenge.

Runs all combinations of:
- Datasets: MUTAG, AIDS, PROTEINS, NCI1, PTC_MR
- Features: topological, ctqw, hybrid
- Kernels: rbf, linear, poly

Generates comprehensive results with statistical analysis.
"""
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json

sys.path.append(str(Path(__file__).parent.parent))

from qkernels.datasets import MolecularGraphDataset
from qkernels.features import (
    TopologicalFeatureExtractor,
    CTQWFeatureExtractor,
    HybridFeatureExtractor,
    extract_features_batch
)
from qkernels.kernels import KernelSVM
from qkernels.eval import ClassificationEvaluator
from qkernels.utils import setup_logging


def run_single_experiment(dataset_name, feature_type, kernel_type, cv_folds=10, random_state=42):
    """
    Run a single experiment configuration.
    
    Returns:
        Dictionary with results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"=" * 80)
    logger.info(f"Running: {dataset_name} + {feature_type} + {kernel_type}")
    logger.info(f"=" * 80)
    
    start_time = time.time()
    
    try:
        # Load dataset
        logger.info(f"Loading {dataset_name} dataset...")
        dataset = MolecularGraphDataset(dataset_name, data_dir='data')
        graphs, labels = dataset.get_graphs_and_labels()
        dataset.summary()
        
        # Extract features
        logger.info(f"Extracting {feature_type} features...")
        
        if feature_type == 'topological':
            extractor = TopologicalFeatureExtractor()
        elif feature_type == 'ctqw':
            extractor = CTQWFeatureExtractor(gamma=1.0, time_points=[0.5, 1.0, 2.0])
        elif feature_type == 'hybrid':
            extractor = HybridFeatureExtractor()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Fit scaler on all data (will be refitted in CV)
        scaler = extractor.fit_scaler(graphs)
        features, feature_names = extract_features_batch(graphs, extractor, scaler)
        
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        # Set up model
        logger.info(f"Setting up {kernel_type} kernel SVM...")
        
        if kernel_type == 'rbf':
            model = KernelSVM(kernel='rbf', C=1.0, gamma='scale')
        elif kernel_type == 'linear':
            model = KernelSVM(kernel='linear', C=1.0)
        elif kernel_type == 'poly':
            model = KernelSVM(kernel='poly', C=1.0, degree=2)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Evaluate with cross-validation
        logger.info(f"Running {cv_folds}-fold cross-validation...")
        evaluator = ClassificationEvaluator(cv_folds=cv_folds, random_state=random_state)
        
        cv_results = evaluator.cross_validate(
            model, features, np.array(labels),
            model_name=f"{dataset_name}_{feature_type}_{kernel_type}"
        )
        
        # Add metadata
        cv_results['dataset'] = dataset_name
        cv_results['feature_type'] = feature_type
        cv_results['kernel_type'] = kernel_type
        cv_results['n_samples'] = len(graphs)
        cv_results['n_features'] = features.shape[1]
        cv_results['n_classes'] = len(set(labels))
        cv_results['feature_names'] = feature_names
        cv_results['total_time'] = time.time() - start_time
        
        logger.info(f"Experiment completed in {cv_results['total_time']:.2f} seconds")
        logger.info(f"Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        logger.info(f"F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        
        return cv_results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return {
            'dataset': dataset_name,
            'feature_type': feature_type,
            'kernel_type': kernel_type,
            'error': str(e),
            'accuracy_mean': np.nan,
            'f1_mean': np.nan
        }


def run_full_benchmark(datasets=None, features=None, kernels=None, 
                      cv_folds=10, random_state=42, output_dir='results'):
    """
    Run complete benchmark across all configurations.
    
    Args:
        datasets: List of dataset names (default: all 5)
        features: List of feature types (default: all)
        kernels: List of kernel types (default: rbf, linear)
        cv_folds: Number of CV folds
        random_state: Random seed
        output_dir: Where to save results
    """
    logger = logging.getLogger(__name__)
    
    # Defaults
    if datasets is None:
        datasets = ['MUTAG', 'AIDS', 'PROTEINS', 'NCI1', 'PTC_MR']
    if features is None:
        features = ['topological', 'ctqw', 'hybrid']
    if kernels is None:
        kernels = ['rbf', 'linear']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("STARTING FULL BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Features: {features}")
    logger.info(f"Kernels: {kernels}")
    logger.info(f"CV Folds: {cv_folds}")
    logger.info(f"Random Seed: {random_state}")
    logger.info("=" * 80)
    
    # Run all combinations
    all_results = []
    summary_results = []
    
    for dataset in datasets:
        for feature in features:
            for kernel in kernels:
                result = run_single_experiment(
                    dataset, feature, kernel,
                    cv_folds=cv_folds,
                    random_state=random_state
                )
                
                all_results.append(result)
                
                # Add to summary
                summary_results.append({
                    'Dataset': dataset,
                    'Features': feature,
                    'Kernel': kernel,
                    'Accuracy': f"{result.get('accuracy_mean', np.nan):.4f}",
                    'Accuracy_Std': f"{result.get('accuracy_std', np.nan):.4f}",
                    'F1-Score': f"{result.get('f1_mean', np.nan):.4f}",
                    'F1_Std': f"{result.get('f1_std', np.nan):.4f}",
                    'Time(s)': f"{result.get('total_time', 0):.2f}"
                })
                
                # Save intermediate results
                results_file = output_path / f"{dataset}_{feature}_{kernel}_results.json"
                with open(results_file, 'w') as f:
                    # Convert numpy types to native Python for JSON serialization
                    serializable_result = {}
                    for key, value in result.items():
                        if isinstance(value, (np.integer, np.floating)):
                            serializable_result[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            serializable_result[key] = value.tolist()
                        elif isinstance(value, dict):
                            serializable_result[key] = {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in value.items()
                            }
                        else:
                            serializable_result[key] = value
                    
                    json.dump(serializable_result, f, indent=2)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_results)
    
    # Save summary
    summary_file = output_path / 'benchmark_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    logger.info("=" * 80)
    logger.info("BENCHMARK COMPLETED")
    logger.info("=" * 80)
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("\nResults Summary:")
    logger.info("\n" + summary_df.to_string(index=False))
    
    # Find best configuration per dataset
    logger.info("\n" + "=" * 80)
    logger.info("BEST CONFIGURATIONS PER DATASET")
    logger.info("=" * 80)
    
    for dataset in datasets:
        dataset_results = summary_df[summary_df['Dataset'] == dataset]
        if len(dataset_results) > 0:
            # Convert accuracy string to float for comparison
            dataset_results['Accuracy_float'] = dataset_results['Accuracy'].astype(float)
            best = dataset_results.loc[dataset_results['Accuracy_float'].idxmax()]
            logger.info(f"\n{dataset}:")
            logger.info(f"  Best: {best['Features']} + {best['Kernel']}")
            logger.info(f"  Accuracy: {best['Accuracy']} ± {best['Accuracy_Std']}")
            logger.info(f"  F1-Score: {best['F1-Score']} ± {best['F1_Std']}")
    
    return summary_df, all_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive benchmark for molecular graph classification'
    )
    
    parser.add_argument('--datasets', nargs='+', 
                       choices=['MUTAG', 'AIDS', 'PROTEINS', 'NCI1', 'PTC_MR'],
                       help='Datasets to benchmark (default: all)')
    parser.add_argument('--features', nargs='+',
                       choices=['topological', 'ctqw', 'hybrid'],
                       help='Feature types to test (default: all)')
    parser.add_argument('--kernels', nargs='+',
                       choices=['rbf', 'linear', 'poly'],
                       help='Kernel types to test (default: rbf, linear)')
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
    setup_logging(log_level)
    
    # Run benchmark
    summary_df, all_results = run_full_benchmark(
        datasets=args.datasets,
        features=args.features,
        kernels=args.kernels,
        cv_folds=args.cv_folds,
        random_state=args.random_seed,
        output_dir=args.output_dir
    )
    
    print("\n✅ Benchmark completed successfully!")
    print(f"📊 Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
