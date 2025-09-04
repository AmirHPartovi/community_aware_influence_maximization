import networkx as nx
from networkx.algorithms.community import girvan_newman


def detect_communities_girvan_newman(G: nx.Graph, level: int = 1):
    """
    Detect communities using Girvan-Newman algorithm.
    level = how many splits (e.g., 1 means one split into 2 communities)
    Returns: dict {node: community_id}
    """
    comp = girvan_newman(G)
    limited = None
    for i in range(level):
        limited = next(comp)
    mapping = {}
    for cid, comm in enumerate(limited):
        for node in comm:
            mapping[node] = cid
    return mapping


if __name__ == "__main__":
    G = nx.karate_club_graph()
    communities = detect_communities_girvan_newman(G, level=2)
    print(communities)
