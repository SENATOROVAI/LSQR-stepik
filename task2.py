import numpy as np
from scipy.sparse.linalg import lsqr

A = np.array([[1., 2.],
              [3., 4.],
              [5., 6.]])
b = np.array([1., 1., 1.])
