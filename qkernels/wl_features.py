"""
Weisfeiler-Lehman Enhanced Features for Graph Classification.

Based on state-of-the-art research:
- Shervashidze et al. (2011): WL kernels achieve best performance
- AERK paper (2023): CTQW + entropy alignment outperforms classical methods
- Graph kernel survey (2019): WL-OA provides highest average accuracy

This module implements advanced WL-based feature extraction combined with CTQW.
"""
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.linalg import expm
from sklearn.preprocessing import StandardScaler
import logging
import hashlib

logger = logging.getLogger(__name__)


class WeisfeilerLehmanFeatureExtractor:
    """
    Weisfeiler-Lehman graph feature extraction with multiple refinement iterations.
    
    The WL algorithm iteratively refines node labels based on neighborhood structure,
    creating a hierarchy of increasingly detailed graph representations.
    """
    
    def __init__(self, h=3):
        """
        Initialize WL feature extractor.
        
        Args:
            h: Number of WL iterations (refinement depth)
        """
        self.h = h
        self.feature_names = []
        
        # Generate feature names for each WL iteration
        for iteration in range(h + 1):
            self.feature_names.append(f'wl_iter_{iteration}_label_count')
            self.feature_names.append(f'wl_iter_{iteration}_unique_labels')
            self.feature_names.append(f'wl_iter_{iteration}_entropy')
    
    def _weisfeiler_lehman_step(self, graph, node_labels):
        """
        Perform one WL refinement step.
        
        Args:
            graph: NetworkX graph
            node_labels: Dictionary mapping nodes to labels
            
        Returns:
            New node labels after refinement
        """
        new_labels = {}
        
        for node in graph.nodes():
            # Get current node label
            current_label = node_labels.get(node, 0)
            
            # Get sorted neighbor labels
            neighbor_labels = sorted([
                node_labels.get(neighbor, 0) 
                for neighbor in graph.neighbors(node)
            ])
            
            # Create new label by hashing: current_label + sorted neighbor labels
            label_string = f"{current_label}_{'_'.join(map(str, neighbor_labels))}"
            new_label = int(hashlib.md5(label_string.encode()).hexdigest()[:8], 16)
            
            new_labels[node] = new_label
        
        return new_labels
    
    def extract_features(self, graph):
        """
        Extract WL features from a graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Initialize node labels (use node attributes if available, else uniform)
        if graph.number_of_nodes() == 0:
            return np.zeros(len(self.feature_names), dtype=np.float32)
        
        node_labels = {}
        for node in graph.nodes():
            # Try to use node attributes, otherwise use degree
            if 'label' in graph.nodes[node]:
                node_labels[node] = graph.nodes[node]['label']
            else:
                node_labels[node] = graph.degree(node)
        
        # Perform h iterations of WL refinement
        for iteration in range(self.h + 1):
            # Extract features from current labeling
            label_values = list(node_labels.values())
            
            # Feature 1: Total number of labels (graph size indicator)
            features.append(len(label_values))
            
            # Feature 2: Number of unique labels (diversity)
            features.append(len(set(label_values)))
            
            # Feature 3: Label distribution entropy
            label_counts = defaultdict(int)
            for label in label_values:
                label_counts[label] += 1
            
            total = len(label_values)
            entropy = 0.0
            for count in label_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p + 1e-12)
            features.append(entropy)
            
            # Refine labels for next iteration (except on last iteration)
            if iteration < self.h:
                node_labels = self._weisfeiler_lehman_step(graph, node_labels)
        
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


class AdvancedCTQWFeatureExtractor:
    """
    Advanced CTQW feature extraction based on AERK (2023) paper.
    
    Key improvements:
    - Multiple time scales for temporal dynamics
    - Shannon entropy for each vertex (quantum information)
    - Averaged mixing matrix features
    - Global coherence measures
    """
    
    def __init__(self, gamma=1.0, time_points=None):
        """
        Initialize advanced CTQW feature extractor.
        
        Args:
            gamma: Hamiltonian scaling parameter
            time_points: List of evolution times (more points = better temporal resolution)
        """
        self.gamma = gamma
        # Use more time points for better temporal dynamics capture
        self.time_points = time_points or [0.3, 0.7, 1.5, 3.0, 6.0]
        self.feature_names = []
        
        # Generate comprehensive feature names
        for t in self.time_points:
            self.feature_names.extend([
                f'shannon_entropy_t{t}',      # Quantum Shannon entropy
                f'avg_return_prob_t{t}',       # Average return probability
                f'trace_real_t{t}',            # Real part of trace
                f'trace_imag_t{t}',            # Imaginary part of trace
                f'coherence_t{t}',             # Quantum coherence
                f'mixing_uniformity_t{t}',     # How uniform is the mixing
            ])
    
    def extract_features(self, graph):
        """
        Extract advanced CTQW features.
        
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
            return np.zeros(len(self.feature_names), dtype=np.float32)
        
        # Hamiltonian: H = -γA (continuous-time quantum walk)
        H = -self.gamma * A
        
        # Initial state: uniform superposition
        psi0 = np.ones(n) / np.sqrt(n)
        
        for t in self.time_points:
            try:
                # Evolution operator: U(t) = exp(-iHt)
                U = expm(-1j * H * t)
                
                # Evolved state
                psi_t = U @ psi0
                
                # Probability distribution
                probs = np.abs(psi_t) ** 2
                
                # 1. Shannon entropy (quantum information measure)
                shannon_ent = -np.sum(probs * np.log2(probs + 1e-12))
                
                # 2. Average return probability
                avg_return = np.mean(probs)
                
                # 3. Trace features (global quantum state)
                trace_complex = np.trace(U)
                trace_real = np.real(trace_complex)
                trace_imag = np.imag(trace_complex)
                
                # 4. Quantum coherence (off-diagonal elements)
                rho = np.outer(psi_t, psi_t.conj())  # Density matrix
                coherence = np.sum(np.abs(rho - np.diag(np.diagonal(rho))))
                
                # 5. Mixing uniformity (how close to uniform distribution)
                uniform_prob = 1.0 / n
                mixing_uniformity = 1.0 - np.sum(np.abs(probs - uniform_prob)) / 2.0
                
                features.extend([
                    shannon_ent,
                    avg_return,
                    trace_real,
                    trace_imag,
                    coherence,
                    mixing_uniformity
                ])
                
            except Exception as e:
                logger.warning(f"Error computing CTQW features for t={t}: {e}")
                features.extend([0, 0, 0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names
    
    def fit_scaler(self, graphs):
        """Fit standard scaler on training data."""
        features = [self.extract_features(g) for g in graphs]
        scaler = StandardScaler()
        scaler.fit(features)
        return scaler


class WLCTQWHybridFeatureExtractor:
    """
    State-of-the-art hybrid feature extractor combining:
    - Weisfeiler-Lehman graph refinement
    - Advanced CTQW with quantum information theory
    - Topological indices
    
    This approach is inspired by the AERK (2023) paper and graph kernel survey best practices.
    """
    
    def __init__(self, wl_iterations=3, ctqw_time_points=None):
        """
        Initialize hybrid extractor.
        
        Args:
            wl_iterations: Number of WL refinement iterations
            ctqw_time_points: Time points for CTQW evolution
        """
        self.wl_extractor = WeisfeilerLehmanFeatureExtractor(h=wl_iterations)
        self.ctqw_extractor = AdvancedCTQWFeatureExtractor(
            gamma=1.0,
            time_points=ctqw_time_points or [0.3, 0.7, 1.5, 3.0, 6.0]
        )
        
        # Combine feature names
        self.feature_names = (
            self.wl_extractor.get_feature_names() +
            self.ctqw_extractor.get_feature_names()
        )
    
    def extract_features(self, graph):
        """Extract hybrid WL+CTQW features."""
        wl_features = self.wl_extractor.extract_features(graph)
        ctqw_features = self.ctqw_extractor.extract_features(graph)
        
        return np.concatenate([wl_features, ctqw_features])
    
    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names
    
    def fit_scaler(self, graphs):
        """Fit standard scaler on training data."""
        features = [self.extract_features(g) for g in graphs]
        scaler = StandardScaler()
        scaler.fit(features)
        return scaler


if __name__ == "__main__":
    # Test feature extractors
    logging.basicConfig(level=logging.INFO)
    
    # Create test graph
    G = nx.karate_club_graph()
    
    # Test WL features
    print("Testing Weisfeiler-Lehman features...")
    wl_extractor = WeisfeilerLehmanFeatureExtractor(h=3)
    wl_feat = wl_extractor.extract_features(G)
    print(f"WL features shape: {wl_feat.shape}")
    print(f"WL features: {wl_feat[:6]}")
    
    # Test advanced CTQW features
    print("\nTesting Advanced CTQW features...")
    ctqw_extractor = AdvancedCTQWFeatureExtractor()
    ctqw_feat = ctqw_extractor.extract_features(G)
    print(f"CTQW features shape: {ctqw_feat.shape}")
    print(f"CTQW features (first 6): {ctqw_feat[:6]}")
    
    # Test hybrid features
    print("\nTesting Hybrid WL+CTQW features...")
    hybrid_extractor = WLCTQWHybridFeatureExtractor(wl_iterations=3)
    hybrid_feat = hybrid_extractor.extract_features(G)
    print(f"Hybrid features shape: {hybrid_feat.shape}")
    print(f"Total features: {len(hybrid_extractor.get_feature_names())}")
