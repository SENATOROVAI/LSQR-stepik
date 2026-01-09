from sklearn.linear_model import Ridge
import numpy as np

A = np.array([[1., 2.],
              [3., 4.],
              [5., 6.]])
b = np.array([1., 1., 1.])

model = Ridge(alpha=0, solver="lsqr", fit_intercept=False)
