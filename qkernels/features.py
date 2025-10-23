"""
Feature extraction for molecular graphs.

Implements topological indices, graph spectral features, and quantum-inspired features.
"""
import numpy as np
import networkx as nx
from scipy.linalg import expm
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class TopologicalFeatureExtractor:
    """Extract topological indices from molecular graphs."""

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = [
            'num_nodes', 'num_edges', 'avg_degree', 'density',
            'wiener_index', 'estrada_index', 'randic_index',
            'spectral_radius', 'energy', 'num_triangles',
            'avg_clustering', 'assortativity', 'diameter'
        ]

    def extract_features(self, graph):
        """
        Extract topological features from a single graph.

        Args:
            graph: NetworkX graph

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Basic graph properties
        n = graph.number_of_nodes()
        m = graph.number_of_edges()

        features.append(n)  # num_nodes
        features.append(m)  # num_edges
        features.append(2 * m / n if n > 0 else 0)  # avg_degree
        features.append(2 * m / (n * (n - 1)) if n > 1 else 0)  # density

        # Topological indices
        try:
            features.append(nx.wiener_index(graph))  # wiener_index
        except:
            features.append(0)

        try:
            # Estrada index
            eigenvals = nx.adjacency_matrix(graph).toarray()
            if eigenvals.size > 0:
                features.append(np.sum(np.exp(eigenvals.diagonal())))
            else:
                features.append(0)
        except:
            features.append(0)

        try:
            # Randić index
            randic = 0
            for u, v in graph.edges():
                du = graph.degree(u)
                dv = graph.degree(v)
                if du > 0 and dv > 0:
                    randic += 1 / np.sqrt(du * dv)
            features.append(randic)
        except:
            features.append(0)

        # Spectral properties
        try:
            eigenvals = nx.adjacency_spectrum(graph)
            if len(eigenvals) > 0:
                features.append(np.max(np.real(eigenvals)))  # spectral_radius
                features.append(np.sum(np.abs(eigenvals)))  # energy
            else:
                features.append(0)
                features.append(0)
        except:
            features.append(0)
            features.append(0)

        # Local properties
        try:
            triangles_dict = nx.triangles(graph)  # Returns dict {node: num_triangles}
            features.append(sum(triangles_dict.values()) / 3)  # Each triangle counted 3 times
        except:
            features.append(0)

        try:
            features.append(nx.average_clustering(graph))  # avg_clustering
        except:
            features.append(0)

        try:
            features.append(nx.degree_assortativity_coefficient(graph))  # assortativity
        except:
            features.append(0)

        try:
            features.append(nx.diameter(graph))  # diameter
        except:
            features.append(0)

        return np.array(features, dtype=np.float32)

    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names

    def fit_scaler(self, graphs):
        """
        Fit standard scaler on training data.

        Args:
            graphs: List of NetworkX graphs

        Returns:
            Fitted StandardScaler
        """
        features = [self.extract_features(g) for g in graphs]
        scaler = StandardScaler()
        scaler.fit(features)
        return scaler


class CTQWFeatureExtractor:
    """Extract features based on Continuous-Time Quantum Walks."""

    def __init__(self, gamma=1.0, time_points=None):
        """
        Initialize CTQW feature extractor.

        Args:
            gamma: Scaling parameter for Hamiltonian
            time_points: List of evolution times to evaluate
        """
        self.gamma = gamma
        self.time_points = time_points or [0.5, 1.0, 2.0]
        self.feature_names = []

        # Generate feature names
        for t in self.time_points:
            self.feature_names.extend([
                f'entropy_t{t}', f'avg_return_t{t}', f'trace_real_t{t}', f'trace_imag_t{t}'
            ])

    def extract_features(self, graph):
        """
        Extract CTQW features from a single graph.

        Args:
            graph: NetworkX graph

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Get adjacency matrix
        A = nx.adjacency_matrix(graph).toarray()
        n = A.shape[0]

        if n == 0:
            # Return zeros for empty graphs
            return np.zeros(len(self.feature_names), dtype=np.float32)

        # Hamiltonian
        H = -self.gamma * A

        # Initial state (uniform superposition)
        psi0 = np.ones(n) / np.sqrt(n)

        for t in self.time_points:
            try:
                # Evolution operator
                U = expm(1j * H * t)

                # Evolved state
                psi_t = U @ psi0

                # Probability distribution
                probs = np.abs(psi_t) ** 2

                # Features
                entropy = -np.sum(probs * np.log2(probs + 1e-12))  # Shannon entropy
                avg_return = np.mean(probs)  # Average return probability
                trace_complex = np.trace(U)
                trace_real = np.real(trace_complex)
                trace_imag = np.imag(trace_complex)

                features.extend([entropy, avg_return, trace_real, trace_imag])

            except Exception as e:
                logger.warning(f"Error computing CTQW features for t={t}: {e}")
                features.extend([0, 0, 0, 0])

        return np.array(features, dtype=np.float32)

    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names

    def fit_scaler(self, graphs):
        """
        Fit standard scaler on training data.

        Args:
            graphs: List of NetworkX graphs

        Returns:
            Fitted StandardScaler
        """
        features = [self.extract_features(g) for g in graphs]
        scaler = StandardScaler()
        scaler.fit(features)
        return scaler


class HybridFeatureExtractor:
    """Combine topological and CTQW features."""

    def __init__(self, topo_extractor=None, ctqw_extractor=None):
        """
        Initialize hybrid feature extractor.

        Args:
            topo_extractor: TopologicalFeatureExtractor instance
            ctqw_extractor: CTQWFeatureExtractor instance
        """
        self.topo_extractor = topo_extractor or TopologicalFeatureExtractor()
        self.ctqw_extractor = ctqw_extractor or CTQWFeatureExtractor()

        # Combine feature names
        self.feature_names = (
            self.topo_extractor.get_feature_names() +
            self.ctqw_extractor.get_feature_names()
        )

    def extract_features(self, graph):
        """
        Extract hybrid features from a single graph.

        Args:
            graph: NetworkX graph

        Returns:
            Feature vector as numpy array
        """
        topo_features = self.topo_extractor.extract_features(graph)
        ctqw_features = self.ctqw_extractor.extract_features(graph)

        return np.concatenate([topo_features, ctqw_features])

    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names

    def fit_scaler(self, graphs):
        """
        Fit standard scaler on training data.

        Args:
            graphs: List of NetworkX graphs

        Returns:
            Fitted StandardScaler
        """
        features = [self.extract_features(g) for g in graphs]
        scaler = StandardScaler()
        scaler.fit(features)
        return scaler


def extract_features_batch(graphs, extractor, scaler=None):
    """
    Extract features from a batch of graphs.

    Args:
        graphs: List of NetworkX graphs
        extractor: Feature extractor instance
        scaler: Optional StandardScaler for normalization

    Returns:
        Feature matrix and feature names
    """
    logger.info(f"Extracting features from {len(graphs)} graphs...")

    features = []
    for graph in graphs:
        feat = extractor.extract_features(graph)
        features.append(feat)

    feature_matrix = np.array(features)

    if scaler is not None:
        feature_matrix = scaler.transform(feature_matrix)

    return feature_matrix, extractor.get_feature_names()


if __name__ == "__main__":
    # Test feature extraction
    logging.basicConfig(level=logging.INFO)

    # Create a simple test graph
    G = nx.path_graph(5)
    G.add_edges_from([(1, 3), (2, 4)])

    # Test topological features
    topo_extractor = TopologicalFeatureExtractor()
    topo_feat = topo_extractor.extract_features(G)
    print(f"Topological features: {topo_feat}")
    print(f"Feature names: {topo_extractor.get_feature_names()}")

    # Test CTQW features
    ctqw_extractor = CTQWFeatureExtractor()
    ctqw_feat = ctqw_extractor.extract_features(G)
    print(f"CTQW features: {ctqw_feat}")
    print(f"CTQW feature names: {ctqw_extractor.get_feature_names()}")

    # Test hybrid features
    hybrid_extractor = HybridFeatureExtractor()
    hybrid_feat = hybrid_extractor.extract_features(G)
    print(f"Hybrid features shape: {hybrid_feat.shape}")
    print(f"All feature names: {hybrid_extractor.get_feature_names()}")
