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


def noobar(itrble, units='', description='', update_every=5, num_iter=None, disable=False):
    """Simple progress bar. tqdm slows down tight loops!"""
    if num_iter is None:
        num_iter = len(itrble)

    intervals = 100 // update_every

    last_update = None
    for i, j in enumerate(itrble):
        yield j

        if disable:
            continue

        p = int((i + 1) / num_iter * intervals) * update_every

        if p != last_update:
            last_update = p
            progress_bar = ('|' * (p // update_every)).ljust(intervals)
            print('{} [{}] {}/{} {}'.format(description, progress_bar, i + 1, num_iter, units))
            clear_output(wait=True)
