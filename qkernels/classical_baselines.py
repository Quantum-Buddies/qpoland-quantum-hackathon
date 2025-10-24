"""
Classical baseline kernels for molecular graph classification comparison.

Implements:
1. Weisfeiler-Lehman subtree kernel (WL)
2. Shortest-path kernel (SP)
3. Graphlet kernel (GK)

These serve as classical baselines to compare against quantum approaches.
"""
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
import logging

logger = logging.getLogger(__name__)

try:
    from grakel.kernels import WeisfeilerLehman, ShortestPath
    from grakel.utils import graph_from_networkx
    GRAKEL_AVAILABLE = True
except ImportError:
    GRAKEL_AVAILABLE = False
    logger.warning("GraKeL not available. Classical baselines will use simplified implementations.")


class WeisfeilerLehmanKernel(BaseEstimator):
    """
    Weisfeiler-Lehman subtree kernel implementation.

    The WL kernel compares graphs by iteratively refining node labels based on
    neighborhood structure, then counts common subtree patterns.
    """

    def __init__(self, h=3, normalize=True):
        """
        Initialize WL kernel.

        Args:
            h: Number of WL iterations (height of subtrees)
            normalize: Whether to normalize kernel values
        """
        self.h = h
        self.normalize = normalize
        self._X_train = None

    def _weisfeiler_lehman_labels(self, graphs, h):
        """
        Compute WL labels for all graphs.

        Args:
            graphs: List of NetworkX graphs
            h: Number of iterations

        Returns:
            Dictionary mapping (graph_idx, node) -> final label
        """
        all_labels = {}

        for graph_idx, graph in enumerate(graphs):
            # Initialize node labels (use node degrees)
            node_labels = {}
            for node in graph.nodes():
                if 'label' in graph.nodes[node]:
                    node_labels[node] = graph.nodes[node]['label']
                else:
                    node_labels[node] = graph.degree(node)

            # WL iterations
            for iteration in range(h):
                new_labels = {}

                for node in graph.nodes():
                    # Sort neighbor labels
                    neighbor_labels = sorted([
                        node_labels.get(neighbor, 0)
                        for neighbor in graph.neighbors(node)
                    ])

                    # Create new label: current + sorted neighbors
                    label_string = f"{node_labels[node]}_{'_'.join(map(str, neighbor_labels))}"
                    new_labels[node] = hash(label_string) % 1000000

                node_labels = new_labels

            # Store final labels for this graph
            for node in graph.nodes():
                all_labels[(graph_idx, node)] = node_labels[node]

        return all_labels

    def _count_common_patterns(self, labels1, labels2, graph1, graph2):
        """
        Count common subtree patterns between two graphs.

        Args:
            labels1, labels2: Label dictionaries for both graphs
            graph1, graph2: NetworkX graphs

        Returns:
            Kernel value (similarity score)
        """
        # Count common label patterns
        patterns1 = defaultdict(int)
        patterns2 = defaultdict(int)

        # Count patterns in graph1
        for node in graph1.nodes():
            label = labels1.get(node, 0)
            patterns1[label] += 1

        # Count patterns in graph2
        for node in graph2.nodes():
            label = labels2.get(node, 0)
            patterns2[label] += 1

        # Compute intersection of pattern counts
        common_patterns = 0
        all_patterns = set(patterns1.keys()) | set(patterns2.keys())

        for pattern in all_patterns:
            common_patterns += min(patterns1[pattern], patterns2[pattern])

        return common_patterns

    def fit(self, X, y=None):
        """Fit the kernel (store training graphs)."""
        self._X_train = X
        return self

    def transform(self, X):
        """Compute kernel matrix."""
        if self._X_train is None:
            self._X_train = X

        return self._compute_kernel_matrix(self._X_train, X)

    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute WL kernel matrix."""
        if X2 is None:
            X2 = X1

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        # Compute WL labels for all graphs
        all_labels = self._weisfeiler_lehman_labels(X1 + X2, self.h)

        for i in range(n1):
            for j in range(n2):
                graph1, graph2 = X1[i], X2[j]

                # Extract labels for this pair
                labels1 = {node: all_labels[(i, node)] for node in graph1.nodes()}
                labels2 = {node: all_labels[(len(X1) + j, node)] for node in graph2.nodes()}

                # Count common patterns
                similarity = self._count_common_patterns(labels1, labels2, graph1, graph2)

                # Normalize by graph sizes
                norm_factor = np.sqrt(len(graph1.nodes()) * len(graph2.nodes()))
                if norm_factor > 0:
                    K[i, j] = similarity / norm_factor
                else:
                    K[i, j] = 1.0 if i == j else 0.0

        return K


class ShortestPathKernel(BaseEstimator):
    """
    Shortest-path kernel for graph classification.

    Computes similarity based on distribution of shortest path lengths
    between all pairs of nodes in the graphs.
    """

    def __init__(self, normalize=True):
        """
        Initialize shortest-path kernel.

        Args:
            normalize: Whether to normalize kernel values
        """
        self.normalize = normalize
        self._X_train = None

    def _compute_path_length_distribution(self, graph):
        """
        Compute distribution of shortest path lengths.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary mapping path_length -> count
        """
        if graph.number_of_nodes() < 2:
            return {0: 1}

        # Compute all-pairs shortest paths
        path_lengths = defaultdict(int)

        for source in graph.nodes():
            lengths = nx.single_source_shortest_path_length(graph, source)

            for target, length in lengths.items():
                if source < target:  # Avoid double counting
                    path_lengths[length] += 1

        return dict(path_lengths)

    def _compute_kernel_value(self, dist1, dist2):
        """
        Compute kernel value between two path length distributions.

        Args:
            dist1, dist2: Path length distributions

        Returns:
            Kernel similarity value
        """
        # Compute histogram intersection
        all_lengths = set(dist1.keys()) | set(dist2.keys())
        intersection = 0
        norm1 = norm2 = 0

        for length in all_lengths:
            count1 = dist1.get(length, 0)
            count2 = dist2.get(length, 0)
            intersection += min(count1, count2)
            norm1 += count1
            norm2 += count2

        if norm1 == 0 or norm2 == 0:
            return 1.0 if norm1 == norm2 else 0.0

        return intersection / np.sqrt(norm1 * norm2)

    def fit(self, X, y=None):
        """Fit the kernel."""
        self._X_train = X
        return self

    def transform(self, X):
        """Compute kernel matrix."""
        if self._X_train is None:
            self._X_train = X

        return self._compute_kernel_matrix(self._X_train, X)

    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute shortest-path kernel matrix."""
        if X2 is None:
            X2 = X1

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        # Compute path length distributions
        distributions = []
        for graph in X1 + X2:
            distributions.append(self._compute_path_length_distribution(graph))

        for i in range(n1):
            for j in range(n2):
                dist1 = distributions[i]
                dist2 = distributions[len(X1) + j]
                K[i, j] = self._compute_kernel_value(dist1, dist2)

        return K


class GraphletKernel(BaseEstimator):
    """
    Graphlet kernel based on small induced subgraphs.

    Counts common 3-4 node graphlet patterns between graphs.
    """

    def __init__(self, k=3, normalize=True):
        """
        Initialize graphlet kernel.

        Args:
            k: Maximum graphlet size (3 or 4)
            normalize: Whether to normalize kernel values
        """
        self.k = k
        self.normalize = normalize
        self._X_train = None

    def _count_triangles(self, graph):
        """Count triangles in graph."""
        triangles = 0
        nodes = list(graph.nodes())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                for k in range(j + 1, len(nodes)):
                    if (graph.has_edge(nodes[i], nodes[j]) and
                        graph.has_edge(nodes[j], nodes[k]) and
                        graph.has_edge(nodes[k], nodes[i])):
                        triangles += 1

        return triangles

    def _count_graphlets(self, graph):
        """
        Count small graphlet patterns.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary of graphlet counts
        """
        graphlets = {
            'triangles': self._count_triangles(graph),
            'edges': graph.number_of_edges(),
            'nodes': graph.number_of_nodes()
        }

        if self.k >= 4:
            # Count 4-node graphlets (simplified)
            graphlets['cliques_4'] = 0
            nodes = list(graph.nodes())

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    for k in range(j + 1, len(nodes)):
                        for l in range(k + 1, len(nodes)):
                            # Check if these 4 nodes form a clique
                            if (graph.has_edge(nodes[i], nodes[j]) and
                                graph.has_edge(nodes[i], nodes[k]) and
                                graph.has_edge(nodes[i], nodes[l]) and
                                graph.has_edge(nodes[j], nodes[k]) and
                                graph.has_edge(nodes[j], nodes[l]) and
                                graph.has_edge(nodes[k], nodes[l])):
                                graphlets['cliques_4'] += 1

        return graphlets

    def _compute_kernel_value(self, graphlets1, graphlets2):
        """
        Compute kernel value between two graphlet count vectors.

        Args:
            graphlets1, graphlets2: Graphlet count dictionaries

        Returns:
            Kernel similarity value
        """
        intersection = 0
        norm1 = norm2 = 0

        for key in set(graphlets1.keys()) | set(graphlets2.keys()):
            count1 = graphlets1.get(key, 0)
            count2 = graphlets2.get(key, 0)
            intersection += min(count1, count2)
            norm1 += count1
            norm2 += count2

        if norm1 == 0 or norm2 == 0:
            return 1.0 if norm1 == norm2 else 0.0

        return intersection / np.sqrt(norm1 * norm2)

    def fit(self, X, y=None):
        """Fit the kernel."""
        self._X_train = X
        return self

    def transform(self, X):
        """Compute kernel matrix."""
        if self._X_train is None:
            self._X_train = X

        return self._compute_kernel_matrix(self._X_train, X)

    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute graphlet kernel matrix."""
        if X2 is None:
            X2 = X1

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        # Compute graphlet counts
        graphlet_counts = []
        for graph in X1 + X2:
            graphlet_counts.append(self._count_graphlets(graph))

        for i in range(n1):
            for j in range(n2):
                counts1 = graphlet_counts[i]
                counts2 = graphlet_counts[len(X1) + j]
                K[i, j] = self._compute_kernel_value(counts1, counts2)

        return K


def create_classical_kernel_matrix(X1, X2=None, kernel_type='wl', **kwargs):
    """
    Convenience function to create classical kernel matrices.

    Args:
        X1: List of NetworkX graphs
        X2: Second list of graphs (optional)
        kernel_type: 'wl', 'sp', or 'graphlet'
        **kwargs: Additional parameters for kernel

    Returns:
        Kernel matrix
    """
    if kernel_type == 'wl':
        kernel = WeisfeilerLehmanKernel(**kwargs)
    elif kernel_type == 'sp':
        kernel = ShortestPathKernel(**kwargs)
    elif kernel_type == 'graphlet':
        kernel = GraphletKernel(**kwargs)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return kernel.fit(X1).transform(X2 if X2 is not None else X1)


if __name__ == "__main__":
    # Test classical baselines
    logging.basicConfig(level=logging.INFO)

    # Create test graphs
    G1 = nx.path_graph(5)
    G2 = nx.cycle_graph(5)
    G3 = nx.complete_graph(4)

    test_graphs = [G1, G2, G3]

    logger.info("Testing classical baseline kernels...")

    # Test WL kernel
    try:
        K_wl = create_classical_kernel_matrix(test_graphs, kernel_type='wl', h=2)
        logger.info(f"WL kernel matrix shape: {K_wl.shape}")
        logger.info(f"WL kernel values:\n{K_wl}")

        # Test SP kernel
        K_sp = create_classical_kernel_matrix(test_graphs, kernel_type='sp')
        logger.info(f"SP kernel matrix shape: {K_sp.shape}")
        logger.info(f"SP kernel values:\n{K_sp}")

        logger.info("Classical baselines working correctly!")

    except Exception as e:
        logger.error(f"Classical baseline test failed: {e}")
