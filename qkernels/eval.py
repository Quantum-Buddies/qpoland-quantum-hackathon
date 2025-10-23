"""
Evaluation utilities for molecular graph classification.

Includes cross-validation, metrics computation, and result analysis.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.base import clone
import logging
from typing import Dict, List, Any
import json
import time

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Comprehensive evaluator for graph classification models."""

    def __init__(self, cv_folds: int = 10, random_state: int = 42):
        """
        Initialize evaluator.

        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}

    def cross_validate(self, model, X, y, model_name: str = "model") -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation.

        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target labels
            model_name: Name for logging

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Starting {self.cv_folds}-fold CV for {model_name}...")

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                            random_state=self.random_state)

        fold_results = {
            'accuracies': [],
            'f1_scores': [],
            'predictions': [],
            'true_labels': [],
            'fit_times': [],
            'predict_times': []
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone model to avoid state carryover
            fold_model = clone(model)

            # Fit model
            start_time = time.time()
            fold_model.fit(X_train, y_train)
            fit_time = time.time() - start_time

            # Predict
            start_time = time.time()
            y_pred = fold_model.predict(X_test)
            predict_time = time.time() - start_time

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            # Store results
            fold_results['accuracies'].append(accuracy)
            fold_results['f1_scores'].append(f1)
            fold_results['predictions'].append(y_pred)
            fold_results['true_labels'].append(y_test)
            fold_results['fit_times'].append(fit_time)
            fold_results['predict_times'].append(predict_time)

            logger.info("Fold {}: Acc={:.4f}, F1={:.4f}, Fit={:.2f}s, Pred={:.3f}s".format(
                fold + 1, accuracy, f1, fit_time, predict_time))

        # Compute summary statistics
        results = {
            'model_name': model_name,
            'cv_folds': self.cv_folds,
            'accuracy_mean': np.mean(fold_results['accuracies']),
            'accuracy_std': np.std(fold_results['accuracies']),
            'f1_mean': np.mean(fold_results['f1_scores']),
            'f1_std': np.std(fold_results['f1_scores']),
            'fit_time_mean': np.mean(fold_results['fit_times']),
            'predict_time_mean': np.mean(fold_results['predict_times']),
            'fold_results': fold_results
        }

        logger.info("{} CV Results - Acc: {:.4f}±{:.4f}, F1: {:.4f}±{:.4f}".format(
            model_name, results['accuracy_mean'], results['accuracy_std'],
            results['f1_mean'], results['f1_std']))

        self.results[model_name] = results
        return results

    def compare_models(self, models_dict: Dict[str, Any], X, y) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.

        Args:
            models_dict: Dictionary of {name: model} pairs
            X: Feature matrix
            y: Target labels

        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing {} models...".format(len(models_dict)))

        comparison_results = []

        for name, model in models_dict.items():
            try:
                results = self.cross_validate(model, X, y, name)

                comparison_results.append({
                    'Model': name,
                    'Accuracy_Mean': results['accuracy_mean'],
                    'Accuracy_Std': results['accuracy_std'],
                    'F1_Mean': results['f1_mean'],
                    'F1_Std': results['f1_std'],
                    'Fit_Time_Mean': results['fit_time_mean'],
                    'Predict_Time_Mean': results['predict_time_mean']
                })

            except Exception as e:
                logger.error("Error evaluating {}: {}".format(name, e))
                comparison_results.append({
                    'Model': name,
                    'Accuracy_Mean': np.nan,
                    'Accuracy_Std': np.nan,
                    'F1_Mean': np.nan,
                    'F1_Std': np.nan,
                    'Fit_Time_Mean': np.nan,
                    'Predict_Time_Mean': np.nan
                })

        return pd.DataFrame(comparison_results)

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for name, results in self.results.items():
            serializable_results[name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[name][key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_results[name][key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_results[name][key] = value

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info("Results saved to {}".format(filepath))

    def load_results(self, filepath: str):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        logger.info("Results loaded from {}".format(filepath))


def print_classification_report(y_true, y_pred, model_name: str = "Model"):
    """Print detailed classification report."""
    print("\n{} Classification Report:".format(model_name))
    print("=" * 50)
    print(classification_report(y_true, y_pred))

    # Additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print("Accuracy: {:.4f}".format(accuracy))
    print("Macro F1-Score: {:.4f}".format(f1))


def bootstrap_confidence_interval(metric_values: List[float], confidence_level: float = 0.95,
                                n_bootstrap: int = 1000) -> tuple:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        metric_values: List of metric values from CV folds
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(metric_values) < 2:
        return (np.nan, np.nan)

    bootstrap_samples = []
    n = len(metric_values)

    np.random.seed(42)  # For reproducibility

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        sample = [metric_values[i] for i in indices]
        bootstrap_samples.append(np.mean(sample))

    # Compute confidence interval
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    lower_bound = np.percentile(bootstrap_samples, lower_percentile)
    upper_bound = np.percentile(bootstrap_samples, upper_percentile)

    return (lower_bound, upper_bound)


if __name__ == "__main__":
    # Test evaluator
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # Multi-class

    from qkernels.kernels import KernelSVM

    # Test models
    models = {
        'RBF_SVM': KernelSVM(kernel='rbf', C=1.0),
        'Linear_SVM': KernelSVM(kernel='linear', C=1.0),
    }

    evaluator = ClassificationEvaluator(cv_folds=5)
    comparison_df = evaluator.compare_models(models, X, y)

    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    # Save results
    evaluator.save_results('test_results.json')
