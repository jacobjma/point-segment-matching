
class BNB(RMSD):
    # TODO: Docstring
    def __init__(self, tol=1e-2, transform='rigid', scale_invariant=True, pivot='cop'):

        super(BNB, self).__init__(transform=transform, scale_invariant=scale_invariant, pivot=pivot)

        self._max_level = max(int(np.ceil(np.log2(2 * np.pi / (tol)))), 1)

        self.precalcs = [('pivot', self._calc_pivot),
                         ('norms', self._calc_norms),
                         ('scale', self._calc_scale)]

    def get_rmsd(self, a, b, precalced):

        p, q = self._get_points(a, b, precalced)

        diagonal = precalced[a]['norms'] ** 2 + np.expand_dims(precalced[b]['norms'] ** 2, 1)
        off_diagonal = precalced[a]['norms'] * np.expand_dims(precalced[b]['norms'], 1)

        # positive clockwise angles
        outer = np.dot(q, p.T)
        cross = np.cross(np.expand_dims(q, axis=1), np.expand_dims(p, axis=0))
        angles = (np.arctan2(cross, outer) + 2 * np.pi) % (2 * np.pi)

        node0 = Node(0, (0, 2 * np.pi))
        node0.eval_bound(angles, diagonal, off_diagonal)

        heap = [node0]
        while len(heap) > 0:

            node = heapq.heappop(heap)

            if node.level == self._max_level:
                break
            elif node.level < self._max_level:
                children = node.generate_children(2)
            else:
                continue

            for child in children:
                child.eval_bound(angles, diagonal, off_diagonal)

                heapq.heappush(heap, child)

        lower_bound, permutation = node.eval_bound(angles, diagonal,
                                                   off_diagonal, assignment='hungarian')

        rmsd = safe_rmsd(q, p[permutation])

        return rmsd  # , permutation


class Node(object):
    # TODO: Docstring
    def __init__(self, level, limits):
        self._level = level
        self._limits = limits
        self._lower_bound = None

    @property
    def limits(self):
        return self._limits

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def level(self):
        return self._level

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._lower_bound == other._lower_bound
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self._lower_bound < other._lower_bound
        return NotImplemented

    def eval_bound(self, angles, diagonal, off_diagonal,
                   assignment='nearest'):

        inside = np.logical_and(angles > self.limits[0],
                                angles <= self.limits[1])

        max_cos = np.ones_like(angles)
        max_cos[inside == False] = np.cos(angles[inside == False] - self.limits[1])

        distance = diagonal - 2 * off_diagonal * max_cos

        if assignment.lower() == 'nearest':
            col = np.argmin(distance, axis=0)
            self._lower_bound = distance[range(len(col)), col].sum()

        elif assignment.lower() == 'hungarian':
            row, col = linear_sum_assignment(distance)
            self._lower_bound = distance[row, col].sum()

        else:
            raise NotImplementedError

        return self._lower_bound, col

    def generate_children(self, n):

        sweep = (self.limits[1] - self.limits[0]) / n

        children = []
        for i in range(n):
            child = Node(self._level + 1,
                         (self.limits[0] + i * sweep,
                          self.limits[0] + (i + 1) * sweep))

            children += [child]

        return children
