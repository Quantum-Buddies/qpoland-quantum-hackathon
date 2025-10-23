"""
Quick Start Script for QPoland Hackathon Challenge

Demonstrates the complete pipeline on MUTAG dataset.
Run this first to validate your setup!
"""
import sys
from pathlib import Path
import logging
import numpy as np

# Add qkernels to path
sys.path.append(str(Path(__file__).parent))

from qkernels.datasets import MolecularGraphDataset
from qkernels.features import (
    TopologicalFeatureExtractor,
    CTQWFeatureExtractor,
    HybridFeatureExtractor
)
from qkernels.kernels import KernelSVM
from qkernels.eval import ClassificationEvaluator
from qkernels.utils import setup_logging, print_results_table

def main():
    """Run quick start demo."""
    # Setup logging
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("QPoland Hackathon - Quick Start Demo")
    logger.info("=" * 80)
    
    # 1. Load MUTAG dataset
    logger.info("\n📦 Step 1: Loading MUTAG dataset...")
    dataset = MolecularGraphDataset('MUTAG', data_dir='data')
    graphs, labels = dataset.get_graphs_and_labels()
    dataset.summary()
    
    # 2. Extract features
    logger.info("\n🔬 Step 2: Extracting features...")
    
    # Try different feature extractors
    extractors = {
        'Topological': TopologicalFeatureExtractor(),
        'CTQW': CTQWFeatureExtractor(gamma=1.0, time_points=[0.5, 1.0, 2.0]),
        'Hybrid': HybridFeatureExtractor()
    }
    
    results = {}
    
    for name, extractor in extractors.items():
        logger.info(f"\n--- Testing {name} Features ---")
        
        try:
            # Extract features
            scaler = extractor.fit_scaler(graphs)
            features = np.array([extractor.extract_features(g) for g in graphs])
            features_scaled = scaler.transform(features)
            
            logger.info(f"Feature shape: {features_scaled.shape}")
            logger.info(f"Number of features: {len(extractor.get_feature_names())}")
            
            # 3. Train and evaluate SVM with RBF kernel
            logger.info(f"\n🎯 Step 3: Training SVM with RBF kernel...")
            
            model = KernelSVM(kernel='rbf', C=1.0, gamma='scale')
            evaluator = ClassificationEvaluator(cv_folds=10, random_state=42)
            
            cv_results = evaluator.cross_validate(
                model, features_scaled, np.array(labels),
                model_name=f"{name}_RBF"
            )
            
            results[name] = cv_results
            
            logger.info(f"✅ {name} Features:")
            logger.info(f"   Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            logger.info(f"   F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        
        except Exception as e:
            logger.error(f"❌ {name} Features failed: {e}")
            continue
    
    # 4. Print summary
    logger.info("\n" + "=" * 80)
    logger.info("QUICK START DEMO COMPLETED")
    logger.info("=" * 80)
    
    if results:
        print_results_table(results)
        
        # Find best method
        best_method = max(results.items(), key=lambda x: x[1]['accuracy_mean'])
        logger.info(f"\n🏆 Best Method: {best_method[0]}")
        logger.info(f"   Accuracy: {best_method[1]['accuracy_mean']:.4f} ± {best_method[1]['accuracy_std']:.4f}")
        logger.info(f"   F1-Score: {best_method[1]['f1_mean']:.4f} ± {best_method[1]['f1_std']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Next Steps:")
    logger.info("1. Run on all datasets: python experiments/run_full_benchmark.py")
    logger.info("2. Try quantum kernels: python experiments/run_cv.py --dataset MUTAG --feature quantum --kernel quantum")
    logger.info("3. Optimize hyperparameters for best performance")
    logger.info("4. Generate visualizations and write report")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
