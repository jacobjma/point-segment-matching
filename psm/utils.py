import numpy as np


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

