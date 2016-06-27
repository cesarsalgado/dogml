import numpy as np


class Whitening:
    def __init__(self, zca=True, epsilon=1e-10):
        self.zca = zca
        self.epsilon = epsilon

    # supposing X is already mean centered.
    def fit(self, X):
        #assert np.allclose(X.mean(axis=0), 0.0)
        n = X.shape[0]
        C = np.dot(X.T, X)/float(n)
        self.U, self.s, self.Ut = np.linalg.svd(C)
        assert np.allclose(self.U, self.Ut.T)
        assert np.allclose(np.linalg.norm(self.U, axis=0), 1.0)

    def transform(self, X):
        Z = np.dot(X, self.U)/np.sqrt(self.s + self.epsilon).reshape(1, self.s.size)
        if self.zca:
            Z = np.dot(Z, self.Ut)
        return Z

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
