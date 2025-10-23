"""
Utility functions for quantum molecular graph classification.

Includes logging setup, caching, and helper functions.
"""
import logging
import sys
from pathlib import Path
import pickle
import hashlib
import json
import numpy as np


def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional file to write logs to
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Reduce noise from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def cache_features(cache_dir='cache'):
    """
    Decorator to cache extracted features.
    
    Usage:
        @cache_features()
        def extract_features(graphs):
            # expensive computation
            return features
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create hash of inputs
            key = hashlib.md5(
                json.dumps(str(args) + str(kwargs)).encode()
            ).hexdigest()
            
            cache_file = cache_path / f"{func.__name__}_{key}.pkl"
            
            # Try to load from cache
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass
            
            # Compute and cache
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except:
                pass
            
            return result
        
        return wrapper
    return decorator


def compute_kernel_matrix_stats(K):
    """
    Compute statistics of a kernel matrix.
    
    Args:
        K: Kernel matrix (n x n)
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'mean': np.mean(K),
        'std': np.std(K),
        'min': np.min(K),
        'max': np.max(K),
        'median': np.median(K),
        'trace': np.trace(K),
        'rank': np.linalg.matrix_rank(K),
        'condition_number': np.linalg.cond(K)
    }
    
    # Eigenvalue spectrum
    eigenvals = np.linalg.eigvalsh(K)
    stats['eigenvalue_max'] = np.max(eigenvals)
    stats['eigenvalue_min'] = np.min(eigenvals)
    stats['eigenvalue_mean'] = np.mean(eigenvals)
    stats['effective_dimension'] = np.sum(eigenvals)**2 / np.sum(eigenvals**2)
    
    return stats


def compute_feature_importance(X, y, feature_names, method='mutual_info'):
    """
    Compute feature importance scores.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        method: 'mutual_info' or 'random_forest'
    
    Returns:
        DataFrame with feature importance
    """
    import pandas as pd
    
    if method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_classif
        scores = mutual_info_classif(X, y, random_state=42)
    
    elif method == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        scores = rf.feature_importances_
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': scores
    }).sort_values('importance', ascending=False)
    
    return importance_df


def print_results_table(results_dict, metrics=['accuracy_mean', 'f1_mean']):
    """
    Print results in a nice table format.
    
    Args:
        results_dict: Dictionary of {name: results} pairs
        metrics: List of metrics to display
    """
    import pandas as pd
    
    rows = []
    for name, results in results_dict.items():
        row = {'Method': name}
        for metric in metrics:
            if metric in results:
                row[metric] = f"{results[metric]:.4f}"
                if f"{metric.replace('_mean', '_std')}" in results:
                    row[metric] += f" ± {results[metric.replace('_mean', '_std')]:.4f}"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def save_experiment_config(config, filepath):
    """
    Save experiment configuration to JSON.
    
    Args:
        config: Dictionary with configuration
        filepath: Where to save
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_experiment_config(filepath):
    """
    Load experiment configuration from JSON.
    
    Args:
        filepath: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class ProgressTracker:
    """Simple progress tracker for long-running experiments."""
    
    def __init__(self, total, desc="Progress"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            desc: Description
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.logger = logging.getLogger(__name__)
    
    def update(self, n=1):
        """Update progress by n items."""
        self.current += n
        pct = 100 * self.current / self.total
        self.logger.info(f"{self.desc}: {self.current}/{self.total} ({pct:.1f}%)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.logger.info(f"{self.desc}: Completed!")


if __name__ == "__main__":
    # Test utilities
    setup_logging(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("Testing utilities...")
    
    # Test progress tracker
    with ProgressTracker(10, "Test") as progress:
        for i in range(10):
            progress.update()
    
    logger.info("All tests passed!")
