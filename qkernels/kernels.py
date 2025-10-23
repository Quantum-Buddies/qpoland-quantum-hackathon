"""
Kernel implementations for molecular graph classification.

Includes classical kernels (RBF, polynomial) and quantum-inspired kernels.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class KernelSVM:
    """SVM classifier with custom kernel support."""

    def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3):
        """
        Initialize Kernel SVM.

        Args:
            kernel: Kernel type ('rbf', 'poly', 'linear', 'precomputed')
            C: Regularization parameter
            gamma: Kernel coefficient for RBF
            degree: Degree for polynomial kernel
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.svm = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fit SVM on training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize SVM
        if self.kernel == 'precomputed':
            self.svm = SVC(kernel=self.kernel, C=self.C)
        else:
            self.svm = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree)

        self.svm.fit(X_scaled, y)
        return self

    def predict(self, X):
        """Predict on test data."""
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)

    def predict_proba(self, X):
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.svm.predict_proba(X_scaled)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'degree': self.degree
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class TopologicalKernelSVM(KernelSVM):
    """SVM with Linear Combination of Topological Kernels (LCTK)."""

    def __init__(self, weights=None, **kwargs):
        """
        Initialize LCTK SVM.

        Args:
            weights: Weights for different feature groups
            **kwargs: Arguments passed to KernelSVM
        """
        super().__init__(**kwargs)
        self.weights = weights

    def _compute_lctk_kernel(self, X, Y=None):
        """
        Compute Linear Combination of Topological Kernels.

        Args:
            X: Feature matrix
            Y: Optional second feature matrix

        Returns:
            Kernel matrix
        """
        if Y is None:
            Y = X

        # Define feature groups (based on our feature extractor)
        n_topo = 13  # Number of topological features
        n_ctqw = 12  # Number of CTQW features (4 per time point × 3 time points)

        # Default weights if not provided
        if self.weights is None:
            self.weights = {
                'topo': 0.6,
                'ctqw': 0.4
            }

        # Compute RBF kernels for each group
        K_topo = rbf_kernel(X[:, :n_topo], Y[:, :n_topo] if Y is not None else Y[:, :n_topo])
        K_ctqw = rbf_kernel(X[:, n_topo:n_topo+n_ctqw],
                          Y[:, n_topo:n_topo+n_ctqw] if Y is not None else Y[:, n_topo:n_topo+n_ctqw])

        # Linear combination
        K_combined = self.weights['topo'] * K_topo + self.weights['ctqw'] * K_ctqw

        return K_combined


class QuantumKernelSVM(KernelSVM):
    """SVM with quantum kernel matrix."""

    def __init__(self, quantum_feature_map=None, **kwargs):
        """
        Initialize quantum kernel SVM.

        Args:
            quantum_feature_map: Function to compute quantum features/kernels
            **kwargs: Arguments passed to KernelSVM
        """
        super().__init__(kernel='precomputed', **kwargs)
        self.quantum_feature_map = quantum_feature_map

    def fit(self, X, y, compute_kernel_matrix=True):
        """
        Fit quantum kernel SVM.

        Args:
            X: Original feature matrix (for quantum feature extraction)
            y: Target labels
            compute_kernel_matrix: Whether to precompute full kernel matrix
        """
        if self.quantum_feature_map is None:
            raise ValueError("Quantum feature map must be provided")

        # Compute quantum kernel matrix
        if compute_kernel_matrix:
            logger.info("Computing quantum kernel matrix...")
            K_train = self.quantum_feature_map(X, X)
            self.kernel_matrix = K_train
        else:
            # For large datasets, compute kernel matrix on-the-fly during fitting
            self.kernel_matrix = None

        # Fit SVM with precomputed kernel
        self.svm = SVC(kernel='precomputed', C=self.C)
        self.svm.fit(K_train, y)
        return self

    def predict(self, X):
        """Predict using quantum kernel."""
        if self.kernel_matrix is None:
            raise ValueError("Kernel matrix must be computed for prediction")

        # Compute test kernel matrix
        K_test = self.quantum_feature_map(X, self.X_train_)

        return self.svm.predict(K_test)


def cross_validate_kernel_svm(X, y, kernel_svm, cv=10, random_state=42):
    """
    Perform cross-validation with kernel SVM.

    Args:
        X: Feature matrix
        y: Target labels
        kernel_svm: KernelSVM instance
        cv: Number of folds
        random_state: Random seed

    Returns:
        Dictionary with CV results
    """
    logger.info(f"Performing {cv}-fold CV with {kernel_svm.kernel} kernel...")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    accuracies = []
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit and predict
        kernel_svm.fit(X_train, y_train)
        y_pred = kernel_svm.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        accuracies.append(acc)
        f1_scores.append(f1)

        logger.info("Fold {}: Acc={:.4f}, F1={:.4f}".format(fold+1, acc, f1))

    results = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'fold_results': {
            'accuracies': accuracies,
            'f1_scores': f1_scores
        }
    }

    logger.info("CV Results - Acc: {:.4f}±{:.4f}, F1: {:.4f}±{:.4f}".format(
        results['accuracy_mean'], results['accuracy_std'],
        results['f1_mean'], results['f1_std']))

    return results


def compare_kernels(X, y, cv=10, random_state=42):
    """
    Compare different kernel methods.

    Args:
        X: Feature matrix
        y: Target labels
        cv: Number of folds
        random_state: Random seed

    Returns:
        Dictionary with results for each kernel
    """
    logger.info("Comparing different kernel methods...")

    # Define kernel configurations
    kernels_to_test = [
        ('rbf', KernelSVM(kernel='rbf', C=1.0, gamma='scale')),
        ('linear', KernelSVM(kernel='linear', C=1.0)),
        ('poly_d2', KernelSVM(kernel='poly', C=1.0, degree=2)),
        ('poly_d3', KernelSVM(kernel='poly', C=1.0, degree=3)),
        ('lctk', TopologicalKernelSVM(kernel='precomputed', C=1.0))
    ]

    results = {}

    for name, kernel_svm in kernels_to_test:
        try:
            if name == 'lctk':
                # For LCTK, we need to handle the kernel computation differently
                # This is a simplified version - in practice, you'd precompute the kernel matrix
                cv_results = cross_validate_kernel_svm(X, y, kernel_svm, cv, random_state)
            else:
                cv_results = cross_validate_kernel_svm(X, y, kernel_svm, cv, random_state)

            results[name] = cv_results

        except Exception as e:
            logger.error(f"Error testing {name} kernel: {e}")
            results[name] = None

    return results


if __name__ == "__main__":
    # Test kernel implementations
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 100
    n_features = 25  # Hybrid features (13 topo + 12 ctqw)

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    # Test basic kernel SVM
    svm = KernelSVM(kernel='rbf', C=1.0)
    svm.fit(X, y)
    y_pred = svm.predict(X)
    print("RBF SVM training accuracy: {:.3f}".format(np.mean(y_pred == y)))

    # Test cross-validation
    results = cross_validate_kernel_svm(X, y, svm, cv=5)
    print("CV Results: {}".format(results))
