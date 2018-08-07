import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import SubplotBase
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from psm.graph.graphutils import adjacency2edges


def add_colorbar(mapable, ax, position='right', size='5%', pad=0.05, **kwargs):
    if position in ['right', 'left']:
        orientation = 'vertical'
    elif position in ['top', 'bottom']:
        orientation = 'horizontal'

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    cbar = plt.colorbar(mapable, cax=cax, orientation=orientation, **kwargs)
    return cbar


def add_patches(points, segments, ax, c=None, clim=None, **kwargs):
    if c is None:
        colors = ax._get_patches_for_fill.get_next_color()
        c_nans = np.zeros(len(segments), dtype=bool)
    else:
        try:
            colors = mcolors.to_rgba_array(c)
            c_nans = np.zeros(len(segments), dtype=bool)
        except:
            c = np.array(c, dtype=float)
            if not c.shape == (len(segments),):
                raise ValueError()
            colors = None
            c_nans = np.isnan(c)

    patches = []
    for segment, c_nan in zip(segments, c_nans):
        if not c_nan:
            patches.append(Polygon(points[segment], True))

    p = PatchCollection(patches, facecolors=colors, **kwargs)

    ax.add_collection(p)

    if colors is None:
        p.set_array(c[c_nans == 0])

        if clim is not None:
            p.set_clim(clim)

    return p


def graph_embedding(points, adjacency, ax=None, color='k', **kwargs):
    if ax is None:
        ax = plt.subplot()

    edges = adjacency2edges(adjacency)

    points = points[np.array(list(edges)).flatten()]

    p = [[points[:, 0][i:i + 2], points[:, 1][i:i + 2], '-'] for i in range(0, len(points), 2)]
    p = [item for sublist in p for item in sublist]

    ax.plot(*p, color=color, **kwargs)

    return ax


def show_structures(structures, axes=None, nrows=1, n=None, show_order=False):
    if isinstance(axes, SubplotBase):
        n = 1
    if (n is None) & (axes is None):
        n = len(structures)
    elif n is None:
        n = len(axes)

    ncols = -(-n // nrows)

    if axes is None:
        fig, axes = plt.subplots(nrows, ncols)

    if (nrows == 1) & (ncols == 1):
        axes = np.array([axes])
    else:
        axes = axes.ravel()

    for ax, structure in zip(axes, structures[:n]):

        graph_embedding(structure.points, structure.adjacency, ax)

        if show_order:
            for i, point in enumerate(structure.points):
                ax.annotate('{}'.format(i), xy=point)

    return axes
