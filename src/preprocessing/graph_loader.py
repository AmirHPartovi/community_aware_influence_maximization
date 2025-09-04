import networkx as nx
import os


def load_graph(dataset_name: str, dataset_dir: str = "../datasets/") -> nx.Graph:
    """
    Load SNAP dataset into a NetworkX graph.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset file (e.g. "facebook_combined.txt")
    dataset_dir : str
        Directory containing dataset files

    Returns
    -------
    G : networkx.Graph
        Graph object created from the dataset
    """
    file_path = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Try edge list format (SNAP datasets usually are in edge list)
    try:
        G = nx.read_edgelist(file_path, nodetype=int)
        print(f"[INFO] Graph loaded: {dataset_name}")
        print(
            f"[INFO] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G
    except Exception as e:
        raise RuntimeError(f"Could not load dataset {dataset_name}: {e}")


def load_all_snap_graphs(dataset_dir: str = "../datasets/") -> dict:
    """
    Load all graphs from the datasets folder.

    Returns
    -------
    graphs : dict
        Dictionary {dataset_name: networkx.Graph}
    """
    graphs = {}
    for dataset_name in os.listdir(dataset_dir):
        if dataset_name.endswith(".txt") or dataset_name.endswith(".csv"):
            try:
                graphs[dataset_name] = load_graph(dataset_name, dataset_dir)
            except Exception as e:
                print(f"[WARNING] Skipping {dataset_name}: {e}")
    return graphs


if __name__ == "__main__":
    # Example usage
    graphs = load_all_snap_graphs("../datasets/")
    for name, G in graphs.items():
        print(f"{name} -> {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
