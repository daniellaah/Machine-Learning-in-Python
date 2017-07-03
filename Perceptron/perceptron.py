import random
import numpy as np
from sklearn.metrics import accuracy_score
class Perceptron(object):
    def __init__(self):
        self.weight = None

    def fit(self, X, y, alpha=0.001, max_iter=1000):
        X, y = np.array(X), np.array(y)
        sample_nums = X.shape[0]
        X = np.column_stack((np.ones(sample_nums), X))
        feature_nums = X.shape[1]
        self.weight = np.zeros((feature_nums, 1))
        idx = np.array([i for i in range(sample_nums)])
        mis_idx = idx[(np.dot(X, self.weight).reshape(sample_nums,) * y) <= 0]
        iteration = 0
        while(len(mis_idx) > 0 and iteration < max_iter):
            rand_mis_idx = random.choice(mis_idx)
            self.weight += (alpha * y[rand_mis_idx] * X[rand_mis_idx]).reshape(feature_nums, 1)
            mis_idx = idx[(np.dot(X, self.weight).reshape(sample_nums,) * y) <= 0]
            iteration += 1
        return self

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = np.concatenate([np.ones(1), X])
            return 1 if X.dot(self.weight)[0] >= 0 else 0
        sample_nums = X.shape[0]
        X = np.column_stack((np.ones(sample_nums), X))
        return np.apply_along_axis(lambda x: 1 if x >= 0 else -1, 1, np.dot(X, self.weight))

    def score(self, X, y):
        y_predict = self.predict(X)
        return accuracy_score(y_predict, y)
