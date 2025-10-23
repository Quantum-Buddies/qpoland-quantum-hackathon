"""
Test improved WL+CTQW features vs baseline features on MUTAG.

Based on state-of-the-art research showing WL-OA achieves best performance.
"""
import sys
import numpy as np
from pathlib import Path

# Add qkernels to path
sys.path.append(str(Path(__file__).parent))

from qkernels.datasets import MolecularGraphDataset
from qkernels.features import TopologicalFeatureExtractor, CTQWFeatureExtractor, HybridFeatureExtractor
from qkernels.wl_features import WeisfeilerLehmanFeatureExtractor, AdvancedCTQWFeatureExtractor, WLCTQWHybridFeatureExtractor
from qkernels.kernels import KernelSVM
from qkernels.eval import ClassificationEvaluator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_features(dataset_name='MUTAG', cv_folds=10):
    """Compare different feature extraction methods."""
    
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = MolecularGraphDataset(dataset_name, data_dir='data')
    graphs = dataset.graphs
    labels = np.array(dataset.labels)
    
    logger.info(f"Loaded {len(graphs)} graphs with {len(set(labels))} classes")
    
    # Define feature extractors to test
    extractors = {
        'Baseline_Topological': TopologicalFeatureExtractor(),
        'Baseline_CTQW': CTQWFeatureExtractor(),
        'Baseline_Hybrid': HybridFeatureExtractor(),
        'WL_Only': WeisfeilerLehmanFeatureExtractor(h=3),
        'Advanced_CTQW': AdvancedCTQWFeatureExtractor(),
        'WL+CTQW_Hybrid': WLCTQWHybridFeatureExtractor(wl_iterations=3),
    }
    
    results = {}
    
    for name, extractor in extractors.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {name}")
        logger.info(f"{'='*60}")
        
        try:
            # Extract features
            logger.info("Extracting features...")
            features = [extractor.extract_features(g) for g in graphs]
            features = np.array(features)
            
            # Fit scaler
            scaler = extractor.fit_scaler(graphs)
            features_scaled = scaler.transform(features)
            
            logger.info(f"Feature shape: {features_scaled.shape}")
            
            # Train RBF SVM with cross-validation
            logger.info("Training RBF SVM with 10-fold CV...")
            model = KernelSVM(kernel='rbf', C=1.0, gamma='scale')
            evaluator = ClassificationEvaluator(cv_folds=cv_folds, random_state=42)
            
            cv_results = evaluator.cross_validate(model, features_scaled, labels, name)
            
            results[name] = {
                'accuracy': cv_results['accuracy_mean'],
                'accuracy_std': cv_results['accuracy_std'],
                'f1': cv_results['f1_mean'],
                'f1_std': cv_results['f1_std'],
                'n_features': features_scaled.shape[1]
            }
            
            logger.info(f"✓ {name}: Acc={cv_results['accuracy_mean']:.4f}±{cv_results['accuracy_std']:.4f}, F1={cv_results['f1_mean']:.4f}±{cv_results['f1_std']:.4f}")
            
        except Exception as e:
            logger.error(f"✗ Error with {name}: {e}")
            results[name] = None
    
    # Print comparison table
    logger.info(f"\n{'='*80}")
    logger.info("RESULTS COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"{'Method':<25} {'#Features':<12} {'Accuracy':<20} {'F1-Score':<20}")
    logger.info(f"{'-'*80}")
    
    for name, result in results.items():
        if result is not None:
            logger.info(f"{name:<25} {result['n_features']:<12} "
                       f"{result['accuracy']:.4f}±{result['accuracy_std']:.4f}      "
                       f"{result['f1']:.4f}±{result['f1_std']:.4f}")
    
    logger.info(f"{'='*80}")
    
    # Find best method
    best_method = max(results.items(), key=lambda x: x[1]['accuracy'] if x[1] else 0)
    logger.info(f"\n🏆 BEST METHOD: {best_method[0]}")
    logger.info(f"   Accuracy: {best_method[1]['accuracy']:.4f}±{best_method[1]['accuracy_std']:.4f}")
    logger.info(f"   F1-Score: {best_method[1]['f1']:.4f}±{best_method[1]['f1_std']:.4f}")
    
    # Calculate improvement over baseline
    if 'Baseline_Hybrid' in results and results['Baseline_Hybrid']:
        baseline_acc = results['Baseline_Hybrid']['accuracy']
        best_acc = best_method[1]['accuracy']
        improvement = ((best_acc - baseline_acc) / baseline_acc) * 100
        logger.info(f"\n📈 IMPROVEMENT OVER BASELINE: {improvement:+.2f}%")
    
    return results


if __name__ == "__main__":
    results = test_features('MUTAG', cv_folds=10)
