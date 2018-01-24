import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.utils import shuffle

class GMM():
    def __init__(self, n_components=2, max_iter=100):
        self.n_comp = 2
        self.max_iter = max_iter
        self.weights_ = []
        self.means_ = []
        self.covariances_ = []

    def fit(self, X):
        m, n = X.shape
        means = [np.random.standard_normal(n) for i in range(self.n_comp)]
        sigmas = [np.identity(n) for i in range(self.n_comp)]
        pis = [1/self.n_comp for i in range(self.n_comp)]
        # EM
        for i in range(self.max_iter):
            # E Step
            predict_gausses = [multivariate_normal(mean, sigma) for mean, sigma in zip(means, sigmas)]
            gauss_sum = 0
            for pi, predict_gauss in zip(pis, predict_gausses):
                gauss_sum += pi * predict_gauss.pdf(X)
            gammas = [pi * predict_gauss.pdf(X) / gauss_sum for pi, predict_gauss in zip(pis, predict_gausses)]

            # M Step
            means = [np.dot(gamma, X) / np.sum(gamma) for gamma in gammas]
            sigmas = [np.dot(gamma * (X - mean).T, X - mean) / np.sum(gamma) for gamma, mean in zip(gammas, means)]
            pis = [np.sum(gamma) / m for gamma in gammas]
        self.weights_ = pis
        self.covariances_ = sigmas
        self.means_ = means
        return self
