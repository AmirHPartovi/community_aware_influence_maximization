import networkx as nx
from networkx.algorithms.community.quality import modularity


def evaluate_modularity(G: nx.Graph, partition: dict) -> float:
    """
    Evaluate modularity of a given partition.
    partition: dict {node: community_id}
    """
    # convert partition dict to list of sets
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, set()).add(node)
    communities = list(communities.values())

    return modularity(G, communities)


if __name__ == "__main__":
    from src.community_detection.louvain import detect_communities_louvain
    G = nx.karate_club_graph()
    partition = detect_communities_louvain(G)
    q = evaluate_modularity(G, partition)
    print(f"Modularity: {q:.4f}")
