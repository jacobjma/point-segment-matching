import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

from psm.build import build_lattice_points, lattice_traversal
from psm.register import RMSD
from psm.structures import traverse_from_all
from psm.graph.geometric import urquhart
from psm.plotutils import add_colorbar


def continuum_strain(points, W, A):
    x = points[:, 0]
    y = points[:, 1]

    w = 1 / W ** 2 / 2

    exx = A * (x + y) * np.exp(-w * (x ** 2 + y ** 2))
    eyy = A * (x - y) * np.exp(-w * (x ** 2 + y ** 2))

    exy = A / 2 * \
          ((np.exp(-w * y ** 2) * (np.sqrt(np.pi) / (2 * np.sqrt(w)) -
                                   np.sqrt(np.pi) * np.sqrt(w) * y ** 2) * erf(np.sqrt(w) * x) +
            y * np.exp(w * (-x ** 2 - y ** 2))) +
           (np.exp(-w * x ** 2) * (np.sqrt(np.pi) / (2 * np.sqrt(w)) -
                                   np.sqrt(np.pi) * np.sqrt(w) * x ** 2) * erf(np.sqrt(w) * y) -
            x * np.exp(w * (-x ** 2 - y ** 2))))

    return exx, eyy, exy


def displacements(points, W, A):
    new_points = points.copy()

    x = points[:, 0]
    y = points[:, 1]

    w = 1 / W ** 2 / 2

    new_points[:, 0] += A * (-np.exp(w * (-x ** 2 - y ** 2)) / (2 * w) +
                             (np.exp(-w * y ** 2) * np.sqrt(np.pi) *
                              y * erf(np.sqrt(w) * x)) / (2 * np.sqrt(w)))

    new_points[:, 1] += A * (np.exp(w * (-x ** 2 - y ** 2)) / (2 * w) +
                             (np.exp(-w * x ** 2) * np.sqrt(np.pi) *
                              x * erf(np.sqrt(w) * y)) / (2 * np.sqrt(w)))

    return new_points


W = 18
A = .008
tol = 1e-4
show_plots = False

a = np.array([0, 1])
b = np.array([1, 0])

points = build_lattice_points(a, b, 60)

points = displacements(points, W, A)

adjacency = urquhart(points)

structures = traverse_from_all(points, adjacency, max_depth=2)

templates = lattice_traversal(a, b, max_depth=2)

rmsd_calc = RMSD(pivot='front')

rmsd_calc.register(templates, structures)

strain, rotation = rmsd_calc.calc_strain(structures)

exx, eyy, exy = strain[:, 0, 0], strain[:, 1, 1], strain[:, 0, 1]

exx_exact, eyy_exact, exy_exact = continuum_strain(structures.fronts, W, A)

if not show_plots:
    assert np.nanmean(np.abs(exx - exx_exact)) < tol
    assert np.nanmean(np.abs(eyy - eyy_exact)) < tol
    assert np.nanmean(np.abs(exy - exy_exact)) < tol
    quit()

fig, axes = plt.subplots(3, 3)

sc = axes[0, 0].scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=exx)
add_colorbar(sc, axes[0, 0], format='%.0e')

sc = axes[0, 1].scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=exx_exact)
add_colorbar(sc, axes[0, 1], format='%.0e')

sc = axes[0, 2].scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=exx_exact - exx)
add_colorbar(sc, axes[0, 2], format='%.0e')

sc = axes[1, 0].scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=eyy)
add_colorbar(sc, axes[1, 0], format='%.0e')

sc = axes[1, 1].scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=eyy_exact)
add_colorbar(sc, axes[1, 1], format='%.0e')

sc = axes[1, 2].scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=eyy_exact - eyy)
add_colorbar(sc, axes[1, 2], format='%.0e')

sc = axes[2, 0].scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=exy)
add_colorbar(sc, axes[2, 0], format='%.0e')

sc = axes[2, 1].scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=exy_exact)
add_colorbar(sc, axes[2, 1], format='%.0e')

sc = axes[2, 2].scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=exy_exact - exy)
add_colorbar(sc, axes[2, 2], format='%.0e')

for ax in axes.ravel():
    ax.axis('equal')
    ax.axis('off')

plt.show()
