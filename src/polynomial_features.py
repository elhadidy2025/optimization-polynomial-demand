#Polynomial feature construction using  NumPy.



from itertools import combinations_with_replacement
import numpy as np


class PolynomialFeatures:
    def __init__(self, degree=3, include_bias=True):
        self.degree = int(degree)
        self.include_bias = include_bias
        self.powers_ = None
        self.n_input_features_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        n_features = X.shape[1]
        powers = []
        if self.include_bias:
            powers.append(tuple())
        for deg in range(1, self.degree + 1):
            powers.extend(combinations_with_replacement(range(n_features), deg))
        self.powers_ = powers
        self.n_input_features_ = n_features
        return self

    def transform(self, X):
        if self.powers_ is None:
            raise RuntimeError("Call fit before transform")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_input_features_:
            raise ValueError("X has a different number of columns than during fit")
        cols = []
        for comb in self.powers_:
            if len(comb) == 0:
                cols.append(np.ones(X.shape[0]))
            else:
                cols.append(np.prod(X[:, comb], axis=1))
        return np.vstack(cols).T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
