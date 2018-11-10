import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import SubplotBase
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi
import scipy.ndimage

from psm.graph.graphutils import edges2adjacency


def interpolate_along_line(points, values, start, end, n):
    # TODO : Implement averaging across a width

    start = np.array(start)
    end = np.array(end)
    t = np.linspace(0, 1, n)
    ptp = np.ptp(points, axis=0)

    dt = np.linalg.norm(end - start, axis=0) / n

    line = start + (np.array(end) - start)[None, :] * t[:, None]

    nx = ptp[0] / dt
    ny = ptp[1] / dt

    interpolated, extent = interpolate(points, values, nx, ny)

    dx = nx / (extent[1] - extent[0])
    dy = ny / (extent[2] - extent[3])

    interpolated = scipy.ndimage.map_coordinates(interpolated, np.vstack((line[:, 0] * dx, line[:, 1] * dy)), order=1)

    return interpolated


def interpolate(points, values, nx, ny, extent=None, method='linear', **kwargs):
    if extent is None:
        xlim = [points[:, 0].min(), points[:, 0].max()]
        ylim = [points[:, 1].max(), points[:, 1].min()]
        extent = xlim + ylim

    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    x, y = np.meshgrid(x, y, indexing='ij')

    interpolated = griddata(points, values, (x, y), method=method, **kwargs)

    return interpolated, extent


def _equation_of_line(p1, p2):
    if p1[0] < p2[0]:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - m * p1[0]

    else:
        m = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p2[1] - m * p2[0]

    return [m, b]


def _halfspace(p1, p2, tol=1e-12):
    vector = p2 - p1

    if np.abs(vector[0]) < tol:
        halfspace = [-1, 0, 0]

    elif np.abs(vector[1]) < tol:
        halfspace = [0, -1, 0]

    else:
        coeff = _equation_of_line(p1, p2)
        if vector[0] > 0:
            halfspace = [-coeff[0], 1, -coeff[1]]
        else:
            halfspace = [coeff[0], -1, coeff[1]]

    return halfspace


def _inscribed_radius(polygon, center, tol=1e-12):
    min_r2 = np.inf

    for j in range(len(polygon)):
        p1 = polygon[j - 1]
        p2 = polygon[j]

        a = p2[1] - p1[1]
        b = p2[0] - p1[0]

        if a + b < tol:
            r2 = np.sum((p1 - center) ** 2)
        else:
            r2 = (a * center[0] - b * center[1] + p2[0] * p1[1] - p2[1] * p1[0]) ** 2 / (a ** 2 + b ** 2)

        min_r2 = min(min_r2, r2)

    return r2


def voronoi_polygons(segments):
    vor = Voronoi(segments.fronts)
    voronoi_neighbors = edges2adjacency(vor.ridge_points, len(vor.points))

    polygons = []
    for i, segment in enumerate(segments):
        hull = segment.points[segment.hull()]

        region = np.array(vor.regions[vor.point_region[i]])
        valid_region = region[region > -1]

        vertices = vor.vertices[valid_region]

        is_in_circle = np.sum((vertices - segment.front) ** 2, axis=1) < _inscribed_radius(hull, segment.front)

        if np.all(is_in_circle) & (-1 not in region):
            polygons.append(vertices)
        else:
            equations = []
            feasible = segment.front + 1e-12 * (segment.center - segment.front)

            for j in range(len(hull)):
                p1 = hull[j - 1]
                p2 = hull[j]
                equations.append(_halfspace(p1, p2))

            for j in voronoi_neighbors[i]:
                t = vor.points[j] - vor.points[i]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                midpoint = vor.points[[i, j]].mean(axis=0)
                far_point = midpoint + n

                equations.append(_halfspace(far_point, midpoint))

            equations = np.array(equations)

            try:
                intersections = HalfspaceIntersection(equations, feasible).intersections
                polygons.append(intersections[ConvexHull(intersections).vertices])
            except:

                polygons.append(np.zeros((1, 2)))

    return polygons


def add_labeled_patches(polygons, labels, ax, cmap, descriptions=None, legend=True, **kwargs):
    if descriptions is None:
        descriptions = {label: label for label in labels}

    proxies = []
    patches = []
    for label, color in zip(np.unique(labels), cmap):
        label_polygons = [polygon for polygon, has_label in zip(polygons, label == labels) if has_label]

        patches.append(add_patches(label_polygons, c=color, ax=ax, **kwargs))

        proxies.append(matplotlib.patches.Patch(color=color, label=descriptions[label]))

    if legend:
        ax.legend(handles=proxies)

    return patches


def _get_colors_array(ax, c, n=None, vmin=None, vmax=None, cmap=None):
    if c is None:
        c = ax._get_patches_for_fill.get_next_color()

    try:
        color = matplotlib.colors.to_rgba_array(c)
    except:

        if cmap is None:
            cmap = matplotlib.cm.get_cmap('viridis')

        elif isinstance(cmap, str):
            cmap = matplotlib.cm.get_cmap(cmap)

        if vmin is None:
            vmin = np.nanmin(c)

        if vmax is None:
            vmax = np.nanmax(c)

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        c = np.array(c, dtype=float)

        valid = np.isnan(c) == 0
        colors = np.zeros((len(c), 4))
        colors[valid] = cmap(norm(c[valid]))
    else:
        colors = np.zeros((n, 4))
        colors[:] = matplotlib.colors.to_rgba_array(color)

    return colors


def add_patches(polygons, ax, c=None, vmin=None, vmax=None, cmap=None, **kwargs):
    colors = _get_colors_array(ax, c, len(polygons), vmin=vmin, vmax=vmax, cmap=cmap)

    patches = []
    for polygon in polygons:
        patches.append(Polygon(polygon, True))

    p = PatchCollection(patches, facecolors=colors, **kwargs)

    ax.add_collection(p)

    return p


def add_edges(points, edges, c, color_mode='edges', cmap=None, ax=None, n=2, vmin=None, vmax=None, **kwargs):
    if ax is None:
        ax = plt.subplot()

    if (n < 2) & (color_mode == 'ends'):
        raise ValueError()

    if color_mode is 'ends':
        if cmap is None:
            cmap = matplotlib.cm.get_cmap('viridis')

        elif isinstance(cmap, str):
            cmap = matplotlib.cm.get_cmap(cmap)

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        colors = np.zeros((n * len(edges), 4))
        lines = np.zeros((n * len(edges), 2, 2))
        for i, edge in enumerate(edges):

            edge = list(edge)

            p1, p2 = points[edge[0]], points[edge[1]]
            c1, c2 = norm(c[edge[0]]), norm(c[edge[1]])

            if np.isnan(c1) & np.isnan(c2):
                colors[i * n:i * n + n] = np.zeros((n, 4))

            elif np.isnan(c1):
                colors[i * n:i * n + n] = cmap(c2)
                colors[i * n:i * n + n, 3] = np.linspace(0, 1, n)

            elif np.isnan(c2):
                colors[i * n:i * n + n] = cmap(c1)
                colors[i * n:i * n + n, 3] = np.linspace(0, 1, n)[::-1]

            else:
                colors[i * n:i * n + n] = cmap(c1 + (c2 - c1) * np.linspace(0, 1, n))

            line = p1[None, :] + (p2 - p1)[None, :] * np.linspace(0, 1, n + 1)[:, None]
            line = line.reshape(-1, 1, 2)
            lines[i * n:i * n + n] = np.concatenate([line[:-1], line[1:]], axis=1)

    elif color_mode is 'edges':
        colors = _get_colors_array(ax, c, len(edges))

        lines = np.zeros((len(edges), 2, 2))
        for i, edge in enumerate(edges):
            lines[i, :] = points[list(edge)]

    else:
        raise ValueError()

    lc = LineCollection(lines, colors=colors, cmap=cmap, **kwargs)
    ax.add_collection(lc)
    ax.autoscale()

    if colors is None:
        lc.set_array(np.array(c))
        lc.set_clim([vmin, vmax])

    return lc, ax


def add_colorbar(ax, cmap, vmin, vmax, n_ticks=5, ticks=None, position='bottom', size='5%', pad=0.05, **kwargs):
    if position in ['right', 'left']:
        orientation = 'vertical'
    elif position in ['top', 'bottom']:
        orientation = 'horizontal'

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    if ticks is None:
        ticks = np.linspace(vmin, vmax, n_ticks)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    return plt.colorbar(sm, cax=cax, orientation=orientation, ticks=ticks, **kwargs)


def voronoi_plot(segments, ax=None, labels=None, c=None, vmin=None, vmax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    polygons = voronoi_polygons(segments)

    if labels is not None:
        p = add_labeled_patches(polygons, labels, ax=ax, **kwargs)
    else:
        p = add_patches(polygons, ax=ax, c=c, vmin=vmin, vmax=vmax, **kwargs)

    ax.autoscale()

    return ax, p


def edge_plot(segments, ax=None, labels=None, c=None, vmin=None, vmax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    if labels is not None:

        p = add_labeled_patches(polygons, labels, ax=ax, **kwargs)
    else:
        edges = segments.edges
        if isinstance(c, dict):
            c = [c[frozenset(edge)] for edge in edges]

        lc, ax = add_edges(segments.points, edges, ax=ax, c=c, vmin=vmin, vmax=vmax, **kwargs)

    ax.axis('equal')
    ax.autoscale()

    return ax, lc


def show_segment(segment, ax=None, c='k', show_order=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    edge_plot(segment, ax, c=c)

    if show_order:
        for i, point in enumerate(segment.points):
            ax.annotate('{}'.format(i), xy=point)

    ax.axis('equal')

    return ax


def show_segments(segments, axes=None, nrows=1, n=None, c='k', show_order=False):
    if isinstance(axes, SubplotBase):
        n = 1
    if (n is None) & (axes is None):
        n = len(segments)
    elif n is None:
        n = len(axes)

    ncols = -(-n // nrows)

    if axes is None:
        fig, axes = plt.subplots(nrows, ncols)

    if (nrows == 1) & (ncols == 1):
        axes = np.array([axes])
    else:
        axes = axes.ravel()

    for ax, segment in zip(axes, segments[:n]):

        edge_plot(segment, ax, c=c)

        if show_order:
            for i, point in enumerate(segment.points):
                ax.annotate('{}'.format(i), xy=point)

        ax.axis('equal')

    return axes
