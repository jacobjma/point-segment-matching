from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules=[
    Extension(
        'ssm.qcp',
        sources=['ssm/qcp.pyx'],
        include_dirs=[np.get_include()]
        )
    ]
    
setup(
    name='ssm',
    version='1.0',
    description='Structural Segment Matching',
    author='Jacob Madsen',
    author_email='jacob.jma@gmail.com',
    packages=['ssm'], #same as name
    install_requires=[], #external packages as dependencies
    ext_modules=cythonize(ext_modules),
)
