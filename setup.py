import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

ext_modules = [
    Extension(
        'psm.geometry.qcp',
        sources=['psm/geometry/qcp.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'psm.graph.traversal',
        sources=['psm/graph/traversal.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'psm.graph.subgraph_isomorphism',
        sources=['psm/graph/subgraph_isomorphism.pyx'],
        include_dirs=[np.get_include()]
    )
]

setup(
    name='psm',
    version='1.0',
    description='point-segment-matching',
    author='Jacob Madsen',
    author_email='jacob.jma@gmail.com',
    packages=find_packages(),  # same as name
    install_requires=[],  # external packages as dependencies
    ext_modules=cythonize(ext_modules),
)
