"""
Dataset loading and preprocessing for molecular graph classification.

Supports loading from TUDataset collection and custom formats.
"""
import os
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
import networkx as nx
from torch_geometric.datasets import TUDataset
from grakel.datasets import fetch_dataset
from grakel.utils import graph_from_networkx
import logging

logger = logging.getLogger(__name__)

# Dataset URLs for direct download (fallback if GraKeL fails)
TUD_DATASET_URLS = {
    'AIDS': 'https://www.chrsmrrs.com/graphkerneldatasets/AIDS.zip',
    'MUTAG': 'https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip',
    'NCI1': 'https://www.chrsmrrs.com/graphkerneldatasets/NCI1.zip',
    'PTC_MR': 'https://www.chrsmrrs.com/graphkerneldatasets/PTC_MR.zip',
    'PROTEINS': 'https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip',
}

class MolecularGraphDataset:
    """Unified interface for molecular graph datasets."""

    def __init__(self, name, data_dir='data', use_grakel=True):
        """
        Initialize dataset loader.

        Args:
            name: Dataset name (AIDS, PROTEINS, NCI1, PTC_MR, MUTAG)
            data_dir: Directory to store downloaded data
            use_grakel: Whether to use GraKeL fetcher (faster, preferred)
        """
        self.name = name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.use_grakel = use_grakel

        # Load dataset
        if use_grakel and name != 'AIDS':  # AIDS not always available in GraKeL
            self._load_grakel()
        else:
            self._load_tudataset()

    def _load_grakel(self):
        """Load dataset using GraKeL."""
        try:
            logger.info(f"Loading {self.name} dataset from GraKeL...")
            graphs, labels = fetch_dataset(
                self.name.lower(),
                verbose=False,
                as_graphs=True
            )

            # Convert GraKeL graphs to NetworkX
            self.graphs = []
            self.labels = []

            for graph, label in zip(graphs, labels):
                nx_graph = nx.Graph()

                # Add edges
                for edge in graph.get_edges():
                    nx_graph.add_edge(edge[0], edge[1])

                # Add node labels if available
                if hasattr(graph, 'node_labels') and graph.node_labels:
                    for node, label in graph.node_labels.items():
                        nx_graph.nodes[node]['label'] = label

                self.graphs.append(nx_graph)
                self.labels.append(label)

            logger.info(f"Loaded {len(self.graphs)} graphs from {self.name}")

        except Exception as e:
            logger.warning(f"GraKeL loading failed: {e}. Falling back to TUDataset...")
            self._load_tudataset()

    def _load_tudataset(self):
        """Load dataset using TUDataset (PyTorch Geometric)."""
        try:
            logger.info(f"Loading {self.name} dataset from TUDataset...")

            # Create TUDataset with appropriate parameters
            if self.name == 'PROTEINS':
                # PROTEINS has node attributes
                dataset = TUDataset(root=str(self.data_dir / 'TUDataset'), name=self.name, use_node_attr=True)
            elif self.name in ['AIDS', 'NCI1']:
                # These datasets may have edge attributes
                dataset = TUDataset(root=str(self.data_dir / 'TUDataset'), name=self.name, use_node_attr=True, use_edge_attr=True)
            else:
                dataset = TUDataset(root=str(self.data_dir / 'TUDataset'), name=self.name)

            self.graphs = []
            self.labels = []

            for data in dataset:
                # Convert PyG data to NetworkX
                nx_graph = nx.Graph()

                # Add edges
                edge_list = data.edge_index.t().tolist()
                nx_graph.add_edges_from(edge_list)

                # Add node features if available
                if hasattr(data, 'x') and data.x is not None and data.x.size(0) > 0:
                    for i in range(data.x.size(0)):
                        if i < len(nx_graph.nodes):
                            nx_graph.nodes[i]['features'] = data.x[i].tolist()
                else:
                    # Add node labels if available
                    if hasattr(data, 'node_labels') and data.node_labels is not None:
                        for i, label in enumerate(data.node_labels):
                            if i < len(nx_graph.nodes):
                                nx_graph.nodes[i]['label'] = label.item()

                self.graphs.append(nx_graph)
                self.labels.append(data.y.item())

            logger.info(f"Loaded {len(self.graphs)} graphs from {self.name}")

        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            logger.error("Trying alternative loading method...")
            self._load_from_files()

    def _load_from_files(self):
        """Load dataset directly from TUDataset files."""
        try:
            logger.info(f"Loading {self.name} from raw files...")

            # Try multiple possible paths
            possible_paths = [
                self.data_dir / 'TUDataset' / self.name / 'raw',
                self.data_dir / 'TUDataset' / 'raw' / self.name,
                self.data_dir / 'TUDataset' / self.name
            ]

            raw_dir = None
            for path in possible_paths:
                if path.exists() and (path / f"{self.name}_A.txt").exists():
                    raw_dir = path
                    break

            if raw_dir is None:
                raise FileNotFoundError(f"Could not find raw files for {self.name}")

            # Check if files exist
            a_file = raw_dir / f"{self.name}_A.txt"
            indicator_file = raw_dir / f"{self.name}_graph_indicator.txt"
            labels_file = raw_dir / f"{self.name}_graph_labels.txt"

            if not all([a_file.exists(), indicator_file.exists(), labels_file.exists()]):
                raise FileNotFoundError(f"Required files not found for {self.name} in {raw_dir}")

            logger.info(f"Found files in {raw_dir}")

            # Load labels (common for both formats)
            labels = []
            with open(labels_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            labels.append(int(float(line)))
                        except ValueError:
                            logger.warning(f"Skipping invalid line {line_num} in {labels_file}: {line}")

            if not labels:
                raise ValueError(f"No valid labels found in {labels_file}")

            logger.info(f"Loaded {len(labels)} labels")

            # Load adjacency matrix (TUDataset format varies - try to detect format)
            first_lines = []
            with open(a_file, 'r') as f:
                for i, line in enumerate(f):
                    if i < 10 and line.strip():
                        first_lines.append(line.strip())

            # Detect format based on first few lines
            sample_line = first_lines[0] if first_lines else ""
            if ',' in sample_line:
                # Format: node1,node2 (edges)
                is_edge_format = True
            else:
                # Format: node_id (nodes)
                is_edge_format = False

            logger.info(f"Detected {'edge' if is_edge_format else 'node'} format for {self.name}")

            if is_edge_format:
                # Load edges
                edges = []
                with open(a_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                parts = line.split(',')
                                if len(parts) >= 2:
                                    u, v = map(int, parts[:2])
                                    edges.append((u-1, v-1))  # Convert to 0-based indexing
                            except ValueError:
                                logger.warning(f"Skipping invalid line {line_num} in {a_file}: {line}")

                if not edges:
                    raise ValueError(f"No valid edges found in {a_file}")

                # Load graph indicators for edges
                graph_ids = []
                with open(indicator_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                graph_ids.append(int(line) - 1)  # Convert to 0-based
                            except ValueError:
                                logger.warning(f"Skipping invalid line {line_num} in {indicator_file}: {line}")

                if len(graph_ids) != len(edges):
                    logger.warning(f"Mismatch: {len(edges)} edges but {len(graph_ids)} graph indicators")

                # Group edges by graph
                from collections import defaultdict
                graph_edges = defaultdict(list)

                for i in range(min(len(edges), len(graph_ids))):
                    graph_id = graph_ids[i]
                    u, v = edges[i]
                    graph_edges[graph_id].append((u, v))

                # Create graphs with edges
                self.graphs = []
                self.labels = []

                for graph_id in range(len(labels)):
                    nx_graph = nx.Graph()
                    if graph_id in graph_edges:
                        nx_graph.add_edges_from(graph_edges[graph_id])

                    self.graphs.append(nx_graph)
                    self.labels.append(labels[graph_id])

            else:
                # Node format (like NCI1)
                nodes = []
                with open(a_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                node_id = int(line)
                                nodes.append(node_id - 1)  # Convert to 0-based
                            except ValueError:
                                logger.warning(f"Skipping invalid line {line_num} in {a_file}: {line}")

                if not nodes:
                    raise ValueError(f"No valid nodes found in {a_file}")

                # Load graph indicators for nodes
                graph_ids = []
                with open(indicator_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                graph_ids.append(int(line) - 1)  # Convert to 0-based
                            except ValueError:
                                logger.warning(f"Skipping invalid line {line_num} in {indicator_file}: {line}")

                if len(graph_ids) != len(nodes):
                    logger.warning(f"Mismatch: {len(nodes)} nodes but {len(graph_ids)} graph indicators")

                # Group nodes by graph (create isolated nodes)
                from collections import defaultdict
                graph_nodes = defaultdict(list)

                for i, node_id in enumerate(nodes):
                    if i < len(graph_ids):
                        graph_id = graph_ids[i]
                        graph_nodes[graph_id].append(node_id)

                # Create graphs with nodes
                self.graphs = []
                self.labels = []

                for graph_id in range(len(labels)):
                    nx_graph = nx.Graph()
                    if graph_id in graph_nodes:
                        for node_id in graph_nodes[graph_id]:
                            nx_graph.add_node(node_id)

                    self.graphs.append(nx_graph)
                    self.labels.append(labels[graph_id])

            logger.info(f"Loaded {len(self.graphs)} graphs from {self.name} files")

        except Exception as e:
            logger.error(f"Failed to load {self.name} from files: {e}")
            raise

    def _download_tudataset(self):
        """Download dataset from TU Dortmund repository."""
        if self.name not in TUD_DATASET_URLS:
            raise ValueError(f"No download URL available for {self.name}")

        url = TUD_DATASET_URLS[self.name]
        zip_path = self.data_dir / f"{self.name}.zip"

        logger.info(f"Downloading {self.name} from {url}...")
        urllib.request.urlretrieve(url, zip_path)

        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir / 'TUDataset' / 'raw')

        # Clean up
        zip_path.unlink()

    def get_graphs_and_labels(self):
        """Return graphs and labels."""
        return self.graphs, self.labels

    def get_class_distribution(self):
        """Return class distribution."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def summary(self):
        """Print dataset summary."""
        logger.info(f"\n{self.name} Dataset Summary:")
        logger.info(f"Number of graphs: {len(self.graphs)}")
        logger.info(f"Number of classes: {len(set(self.labels))}")
        logger.info(f"Class distribution: {self.get_class_distribution()}")

        if self.graphs:
            sizes = [len(g.nodes) for g in self.graphs]
            logger.info("Graph sizes: min={}, max={}, avg={:.2f}".format(min(sizes), max(sizes), np.mean(sizes)))


def load_all_datasets(data_dir='data'):
    """Load all required datasets for the challenge."""
    datasets = ['AIDS', 'PROTEINS', 'NCI1', 'PTC_MR', 'MUTAG']

    loaded_datasets = {}
    for name in datasets:
        try:
            dataset = MolecularGraphDataset(name, data_dir)
            loaded_datasets[name] = dataset
            dataset.summary()
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    return loaded_datasets


if __name__ == "__main__":
    # Test loading MUTAG dataset
    logging.basicConfig(level=logging.INFO)
    dataset = MolecularGraphDataset('MUTAG')
    dataset.summary()
