import networkx as nx


def detect_communities_label_propagation(G: nx.Graph):
    """
    Detect communities using Label Propagation.
    Returns: dict {node: community_id}
    """
    communities = list(
        nx.algorithms.community.label_propagation_communities(G))
    mapping = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            mapping[node] = cid
    return mapping


if __name__ == "__main__":
    G = nx.karate_club_graph()
    communities = detect_communities_label_propagation(G)
    print(communities)
