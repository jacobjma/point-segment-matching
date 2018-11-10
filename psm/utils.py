import numpy as np
from IPython.display import clear_output


def flatten(lists):
    return [item for sublist in lists for item in sublist]


def labels2groups(labels, sort_by_counts=False):
    unique, counts = np.unique(labels, return_counts=True)

    if sort_by_counts:
        unique = unique[np.argsort(counts)]

    groups = []
    for label in unique:
        groups.append(np.where(labels == label)[0])

    return groups


def in_groups(groups):
    return list(set(flatten(groups)))


def relabel_groups(groups):
    mapping = {j: i for (i, j) in enumerate(sorted(list(in_groups(groups))))}
    return [[mapping[i] for i in group] for group in groups]


class ProgressBar(object):

    def __init__(self, num_iter, units='', description='', update_every=5, disable=False):

        self._num_iter = num_iter
        self._units = units
        self._description = description
        self._update_every = update_every
        self._disable = disable

        self._intervals = 100 // update_every
        self._last_update = None

    def print(self, i):

        if not self._disable:
            self._print(i)

    def _print(self, i):

        p = int((i + 1) / self._num_iter * self._intervals) * self._update_every

        if p != self._last_update:
            self._last_update = p
            progress_bar = ('|' * (p // self._update_every)).ljust(self._intervals)
            print('{} [{}] {}/{} {}'.format(self._description, progress_bar, i + 1, self._num_iter, self._units))
            clear_output(wait=True)

def bar(itrble, num_iter=None, **kwargs):
    """Simple progress bar. tqdm slows down tight loops!"""

    if num_iter is None:
        num_iter = len(itrble)

    progress_bar = ProgressBar(num_iter, **kwargs)

    for i, j in enumerate(itrble):
        yield j

        progress_bar.print(i)