# Point Segment Matching
Point segment matching is a method for finding structure in point clouds with repeating structures. The main application is for finding strain in atomic resolution images. A paper describing the technique is forthcoming.

In the image below the method is applied to a set of points representing polycrystalline graphene. The analysis clearly shows compressive strain along the grain boundaries and highlights the rotations of the grains.

<p align="center">
  <img src="https://github.com/jacobjma/point-segment-matching/blob/master/notebooks/abstract.png?raw=true" alt="Polycrystaline graphene"/>
</p>

See [example](https://github.com/jacobjma/point-segment-matching/blob/master/notebooks/poly_graphene_traversal.ipynb) for how the above analysis was done or check out the [tutorial](https://github.com/jacobjma/point-segment-matching/blob/master/notebooks/tutorial_nanowire.ipynb) for more information.

## Installation
Install using pip after cloning the repository from Github:

    git clone https://github.com/jacobjma/point-segment-matching.git
    cd point-segment-matching
    pip install .

## Dependencies
* [NumPy](http://docs.scipy.org/doc/numpy/reference/)
* [matplotlib](http://matplotlib.org/)
* [SciPy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [Python Image Library](https://pillow.readthedocs.io/en/5.0.0/)
* [Cython](http://cython.org/)
* [Jupyter](http://jupyter.org/)
