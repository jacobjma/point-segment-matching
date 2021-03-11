def clockwise_traversal(edge, adjacency, clockwise):

    queue = [edge]
    traversal = [edge[0]]
    marked = set((edge[0],))

    while queue:
        edge = queue.pop(0)

        for i in range(len(adjacency[edge[0]])):
            edge = clockwise[edge]

            if not edge[1] in marked:
                queue.append((edge[1], edge[0]))
                traversal.append(edge[1])
                marked.add(edge[1])

    return traversal

def clockwise_traversal_with_depth(edge, adjacency, clockwise, max_depth=None):

    if max_depth is None:
        max_depth = len(adjacency)

    queue = [edge]
    traversal = [edge[0]]
    depth = [0]
    marked = set((edge[0],))

    while queue:
        edge = queue.pop(0)

        for i in range(len(adjacency[edge[0]])):
            edge = clockwise[edge]

            if not edge[1] in traversal:
                current_depth = depth[traversal.index(edge[0])] + 1

                if current_depth < max_depth:
                    depth.append(current_depth)
                else:
                    break

                queue.append((edge[1], edge[0]))
                traversal.append(edge[1])
                marked.add(edge[1])

    return traversal, depth
