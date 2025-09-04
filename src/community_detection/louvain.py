import networkx as nx
import community.community_louvain as community_louvain


def detect_communities_louvain(G):
    """
    Detect communities using Louvain method.
    Returns: dict {node: community_id}
    """
    # G = preprocess_graph(G.copy())  # Work on a copy with processed weights
    partition = community_louvain.best_partition(G)
    return partition


def preprocess_graph(G):
    """Convert negative weights to positive for community detection"""
    for u, v, w in G.edges(data='weight', default=1.0):
        G[u][v]['weight'] = abs(w)  # Take absolute value of weights
    return G


if __name__ == "__main__":
    G = nx.karate_club_graph()
    communities = detect_communities_louvain(G)
    print(communities)
