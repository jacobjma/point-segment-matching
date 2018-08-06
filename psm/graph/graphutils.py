import numpy as np



def adjacency2edges(adjacency):
    edges = [frozenset(itertools.product([i], vertices)) for i, vertices in enumerate(adjacency)]
    edges = set(_flatten(edges))
    return edges

def edges2adjacency(edges, num_nodes):
    adjacency = [set() for i in range(num_nodes)]
    for edge in edges:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    return adjacency

def adjacency2matrix(adjacency):
    m = np.zeros((len(adjacency),) * 2, dtype=bool)
    for i, adjacent in enumerate(adjacency):
        for j in adjacent:
            m[i, j] = True
            m[j, i] = True
    return m

def subgraph(adjacency, indices):
    indices = list(indices)
    return [set(indices.index(j) for j in adjacency[i] if j in indices) for i in indices]
