import numpy as np
from psm.geometry.rmsd import rmsd_kabsch, rmsd_qcp_slow, rmsd_qcp

# TODO: Why does Cython lose precision?

for i in range(100):
    n = 10

    a = np.random.rand(n, 2)
    a = a - np.mean(a, axis=0)

    b = np.random.rand(n, 2)
    b = b - np.mean(b, axis=0)

    assert np.abs(rmsd_qcp_slow(a, b) - rmsd_kabsch(a, b)) < 1e-12
    assert np.abs(rmsd_qcp(a, b) - rmsd_kabsch(a, b)) < 1e-6
