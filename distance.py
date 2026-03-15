import numpy as np


def distance_matrix(X):
    return np.sqrt(np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=-1))
