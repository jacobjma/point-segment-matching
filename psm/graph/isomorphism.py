import itertools

from psm.graph.graphutils import subgraph
from psm.graph.traversal_slow import clockwise_traversal


def check_isomorphism(adjacency, clockwise, other_adjacency):
    """Returns True if two plane graphs rooted graphs are isomorphs else False"""

    for i in adjacency[0]:
        edge = (0, i)
        traversal = clockwise_traversal(edge, adjacency, clockwise)
        permuted_adjacency = subgraph(adjacency, traversal)

        if permuted_adjacency == other_adjacency:
            return True

    return False


class Traversal(object):
    """Helper class for subgraph isomorphism traversal"""

    def __init__(self, edges, order, marked):
        self.edges = edges

        self.order = order

        self.marked = marked

    def __len__(self):
        return len(self.order)

    @property
    def edge(self):
        return self.edges[0]

    def copy(self):
        return self.__class__(self.edges.copy(), self.order.copy(), self.marked.copy())


def subgraph_isomorphism(adjacency, clockwise, subgraph_adjacency):
    # TODO : Implement in Cython
    """Returns all clockwise ordered subgraph isomorphs.

    Parameters:
    ----------
    adjacency : list of sets
        Adjacency list of larger graph, ordered according to a clockwise traversal
    clockwise : dict
        Dictionary directed edges as keys mapping to the next adjacent edge in the clockwise direction.
    subgraph_adjacency : list of sets
        Adjacency list of smaller graph, ordered according to a clockwise traversal
    """
    outer_queue = []
    for i in adjacency[0]:
        outer_queue.append(Traversal([(0, i)], [], set()))

    subgraph_order = len(subgraph_adjacency)
    subgraph_adjacents = [subgraph_adjacency[n] - set(range(n)) for n in range(subgraph_order)]
    degree = [len(adjacent) for adjacent in adjacency]

    results = []
    while outer_queue:
        traversal = outer_queue.pop(0)
        n = len(traversal)

        if n == subgraph_order:
            if subgraph(adjacency, traversal.order) == subgraph_adjacency:
                results.append(traversal)
            continue

        elif len(traversal.edges) == 0:
            pass

        else:
            edge = traversal.edges.pop(0)

            if not edge[0] in traversal.marked:

                for i in range(degree[edge[0]]):
                    edge = clockwise[edge]
                    if edge[1] not in traversal.marked:
                        traversal.edges.append((edge[1], edge[0]))

                traversal.marked.add(edge[0])
                traversal.order.append(edge[0])

                adjacent = adjacency[edge[0]] - traversal.marked
                discrepancy = len(adjacent) - len(subgraph_adjacents[n])

                if discrepancy > 0:
                    if len(subgraph_adjacents[n]) == 0:
                        traversal.marked.update(adjacent)
                        outer_queue.append(traversal)
                    else:
                        for combination in itertools.combinations(adjacent, discrepancy):
                            outer_queue.append(traversal.copy())
                            outer_queue[-1].marked.update(set(combination))
                elif discrepancy < 0:
                    # outer_queue.append(traversal)
                    pass  # TODO : Is this OK?
                else:
                    outer_queue.append(traversal)
            else:
                outer_queue.append(traversal)

    return results
