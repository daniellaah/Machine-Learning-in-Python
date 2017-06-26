import numpy as np
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
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

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    X = X[y < 2]
    y = y[y < 2]
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = Perceptron1().fit(X, y)
    print(clf.score(X_test, y_test))
