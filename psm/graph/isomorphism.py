import itertools

import numpy as np

from psm.graph.graphutils import adjacency2matrix
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


class Probe(object):

    def __init__(self, edge, traversal, matched, marked, queue):
        self._edge = edge
        self._traversal = traversal
        self._matched = matched
        self._marked = marked
        self._removed = set()
        self._queue = queue

    @property
    def edge(self):
        return self._edge

    @property
    def traversal(self):
        return self._traversal[:self.matched]

    @property
    def matched(self):
        return self._matched

    @property
    def marked(self):
        return self._marked

    @property
    def queue(self):
        return self._queue

    def propagate(self, edge):
        self._traversal[self.matched] = edge[1]
        self._matched += 1
        self._edge = edge
        self._marked.add(edge[1])
        self._queue.append((edge[1], edge[0]))

    def branch(self, edge, marked):
        self._removed.update(marked)
        marked_copy = set(self.marked)
        marked_copy.update(marked)

        return self.__class__(edge, self._traversal.copy(), self.matched, marked_copy, self._queue.copy())


def _initial_probes(adjacency, subgraph_order, subgraph_root_degree):
    probes = []
    k = len(adjacency[0]) - subgraph_root_degree

    if k < 0:
        return probes

    for i in adjacency[0]:
        traversal = np.zeros(subgraph_order, dtype=int)
        probe = Probe((0, i), traversal, 1, set((0,)), [])

        adjacent = adjacency[0].copy()
        adjacent.remove(i)

        for combination in itertools.combinations(adjacent, k):
            child = probe.branch((0, i), combination)
            probes.append(child)

    return probes


def _progate_probe(probe, adjacency, clockwise, subgraph_order):
    edge = probe.edge
    for i in range(len(adjacency[probe.edge[0]])):
        edge = clockwise[edge]
        print(edge,probe.matched)
        if probe.matched == subgraph_order:
            pass
        elif not edge[1] in probe.marked:
            probe.propagate(edge)
            print('propagate')


def _check_probe(probe, adjacency, subgraph_adjacency):
    adjacency = adjacency[probe.traversal[:probe.matched, None],
                          probe.traversal[:probe.matched]]

    subgraph_adjacency = subgraph_adjacency[:probe.matched, :probe.matched]

    return np.all(adjacency == subgraph_adjacency)


def subgraph_isomorphisms(adjacency, clockwise, subgraph_adjacency):
    # TODO: Docstring
    # TODO: Can this be sped up?
    subgraph_order = len(subgraph_adjacency)
    adjacency_matrix = adjacency2matrix(adjacency)
    subgraph_adjacency_matrix = adjacency2matrix(subgraph_adjacency)

    result = []
    outer_queue = _initial_probes(adjacency, subgraph_order, len(subgraph_adjacency[0]))
    outer_queue = [outer_queue[0]]
    while outer_queue:
        print('---')
        for probe in outer_queue:
            print(probe._traversal)
            print(probe._marked)
            print(probe._queue)
        print('---')

        probe = outer_queue.pop(0)

        _progate_probe(probe, adjacency, clockwise, subgraph_order)
        if _check_probe(probe, adjacency_matrix, subgraph_adjacency_matrix):
            if probe.matched == subgraph_order:
                result.append(probe)

            elif len(probe.queue) == 0:
                pass
            else:
                print('***')
                edge = probe.queue.pop(0)
                subgraph_tail = np.nonzero(probe.traversal == edge[0])[0][0]

                adjacent = adjacency[edge[0]].copy() - probe.marked
                subgraph_adjacent = subgraph_adjacency[edge[0]].copy() - set(range(len(probe.marked)))
                print(len(adjacent))
                print(probe.marked)
                k = len(adjacent) - len(subgraph_adjacent)
                #k = len(adjacent) - len(subgraph_adjacency[subgraph_tail]) + 1

                print(edge)

                #k = len(adjacency[edge[0]]) - len(subgraph_adjacency[subgraph_tail])

                print(k)
                if k < 0:
                    pass
                else:
                    #adjacent = adjacency[edge[0]].copy()
                    #adjacent.remove(edge[1])

                    for combination in itertools.combinations(adjacent, k):
                        print()
                        child = probe.branch(edge, combination)
                        outer_queue.append(child)

    return [probe.traversal for probe in result]
