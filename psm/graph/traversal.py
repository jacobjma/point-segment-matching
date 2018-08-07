from tqdm import tqdm_notebook as tqdm

from psm.graph.graphutils import find_clockwise


def breadth_first_traversal(node, adjacency):
    queue = [node]
    traversal = [node]

    while queue:
        node = queue.pop(0)

        for adjacent in adjacency[node]:

            if adjacent not in traversal:
                queue.append(adjacent)
                traversal.append(adjacent)

    return traversal


def clockwise_traversal(edge, adjacency, clockwise):
    degree = [len(adjacent) for adjacent in adjacency]
    queue = [edge]
    traversal = [edge[0]]

    while queue:
        edge = queue.pop(0)

        for i in range(degree[edge[0]]):
            edge = clockwise[edge]

            if not edge[1] in traversal:
                queue.append((edge[1], edge[0]))
                traversal.append(edge[1])

    return traversal


def clockwise_traversal_with_depth(edge, adjacency, clockwise, max_depth=None):
    degree = [len(adjacent) for adjacent in adjacency]

    if max_depth is None:
        max_depth = len(degree)

    queue = [edge]
    traversal = [edge[0]]
    depth = [0]

    while queue:
        edge = queue.pop(0)

        for i in range(degree[edge[0]]):
            edge = clockwise[edge]

            if not edge[1] in traversal:
                current_depth = depth[traversal.index(edge[0])] + 1

                if current_depth < max_depth:
                    depth.append(current_depth)
                else:
                    break

                queue.append((edge[1], edge[0]))
                traversal.append(edge[1])

    return traversal, depth


def traverse_from_all(points, adjacency, second_point='random', max_depth=None, permute=None, progress_bar=False):
    if second_point not in ['random', 'all']:
        raise ValueError()

    clockwise = find_clockwise(points, adjacency)

    traversals = []
    for i, point in enumerate(tqdm(points, unit='traversals', disable=not progress_bar)):
        for j in adjacency[i]:

            edge = (i, j)
            traversal, _ = clockwise_traversal_with_depth(edge, adjacency, clockwise, max_depth=max_depth)

            if permute is not None:
                traversals.append([traversal[k] for k in permute if k < len(traversal)])
            else:
                traversals.append(traversal)

            if second_point == 'random':
                sss
                break

    return traversals
