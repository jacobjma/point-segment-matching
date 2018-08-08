import matplotlib.pyplot as plt
import numpy as np

from psm import plotutils
from psm.build import lattice_traversal
from psm.geometry import transform
from psm.graph import urquhart
from psm.register import RMSD
from psm.structures import traverse_from_all

points = np.load('notebooks/data/poly_graphene.npy')

adjacency = urquhart(points)

structures = traverse_from_all(points, adjacency, max_depth=3)

a = [0, 1]
b = [np.sin(2 / 3 * np.pi), np.cos(2 / 3 * np.pi)]
basis = [[0, 0], [1 / np.sqrt(3), 0]]

templates = lattice_traversal(a, b, basis, max_depth=3, graph_func=urquhart)

rmsd_calc = RMSD(transform='similarity', pivot='cop')

rmsd = rmsd_calc.register(templates, structures)

_, best_rmsd = rmsd_calc.best_matches(structures)

strain, rotation = rmsd_calc.calc_strain(structures)

planar = transform.planar_strain(transform.zero_median(strain))

rotation = (rotation - .4) % (np.pi / 3)

# --- Plot results ---

fig = plt.figure(figsize=(12, 6.5))

ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((2, 4), (0, 2))
ax3 = plt.subplot2grid((2, 4), (0, 3))
ax4 = plt.subplot2grid((2, 4), (1, 2))
ax5 = plt.subplot2grid((2, 4), (1, 3))

pointsize = 20
fontsize = 18

p = ax1.scatter(structures.fronts[:, 0], structures.fronts[:, 1], c='k', s=point_size)
ax1.set_title('Points', fontsize=fontsize)

plotutils.graph_embedding(points, adjacency, ax=ax2)
ax2.set_title('Geometric graph', fontsize=fontsize)

p = ax3.scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=best_rmsd, cmap='viridis', vmin=0, vmax=.04, s=point_size)
ax3.set_title('RMSD', fontsize=fontsize)

p = ax4.scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=planar, cmap='coolwarm', vmin=-.1, vmax=.1, s=point_size)
ax4.set_title('Strain', fontsize=fontsize)

p = ax5.scatter(structures.fronts[:, 0], structures.fronts[:, 1], c=rotation, cmap='hsv', s=point_size)
ax5.set_title('Rotation', fontsize=fontsize)

for ax in (ax1, ax2, ax3, ax4, ax5):
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

plt.savefig('abstract.png')