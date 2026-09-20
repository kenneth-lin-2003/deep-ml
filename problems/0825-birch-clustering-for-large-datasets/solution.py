import numpy as np

def birch_cluster(X, threshold):
    """
    Single-level BIRCH clustering.
    X: array-like of shape (n_samples, n_features)
    threshold: float, max allowed subcluster radius
    Returns: list of centroids (each a list of floats), sorted lexicographically.
    """
    ret = []
    X = np.array(X)
    N = np.array([1])
    LS = np.array([X[0].copy()])
    SS = np.array([X[0] ** 2])
    for i in range(1, len(X)):
        center = LS / N[:,None]
        near = np.argmin(np.linalg.norm(center - X[i], axis=1))
        R = np.sqrt(np.sum((SS[near] + X[i] ** 2) / (N[near]+1) - np.square((LS[near] + X[i]) / (N[near]+1))))
        if R <= threshold:
            N[near] += 1
            LS[near] += X[i]
            SS[near] += X[i] ** 2
        else:
            N = np.append(N, 1)
            LS = np.vstack((LS, X[i]))
            SS = np.vstack((SS, X[i] ** 2))
    ret = (LS / N[:,None]).tolist()
    ret.sort()
    return ret