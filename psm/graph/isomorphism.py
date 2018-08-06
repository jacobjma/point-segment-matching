





def check_isomorphism(adjacency, clockwise, other_adjacency, other_clockwise=None):
    if not other_clockwise is None:
        edges = (0, next(iter(other_adjacency[0])))
        traversal = clockwise_traversal(edges, other_adjacency, other_clockwise)[0]
        other_adjacency = subgraph(other_adjacency, traversal)

    for i in adjacency[0]:
        edge = (0, i)
        traversal = clockwise_traversal(edge, adjacency, clockwise)
        permuted = subgraph(adjacency, traversal)

        if permuted == other_adjacency:
            return True

    return False
