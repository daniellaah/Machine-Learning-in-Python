# Reference: http://daniellaah.github.io/2016/Machine-Learning-Andrew-Ng-My-Notes-Week-3-Logistic-Regression.html
import numpy as np
from mlinpy.utils.functions import sigmoid

class LogisticRegression():
    '''
        X.ndim == 2, X.shape == (n_samples, n_features)
        y.ndim == 1, y.shape == (n_samples, )
        W.ndim == 2, W.shape == (n_features, 1)
        b is scalar
    '''
    def __init__(self, learning_rate=0.01, max_iter=100):
        self.W = None
        self.b = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y, print_cost=False, print_num=100):
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, 1), dtype=float)
        self.b = 0.0
        for i in range(self.max_iter):
            y_hat = self.predict_prob(X)
            cost = -np.sum(y*np.log(y_hat) + (1-y)*(np.log(1-y_hat))) / n_samples
            dW = np.dot((y_hat-y).reshape(n_samples, 1).T, X).T / n_samples
            db = np.sum(y_hat-y) / n_samples
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            if i % print_num == 0 and print_cost:
                print ("Cost after iteration {}: {:.6f}".format(i, cost))
        return self

    def predict_prob(self, X):
        n_samples = X.shape[0]
        return sigmoid(np.dot(X, self.W) + self.b).reshape(n_samples,)

    def predict(self, X, threshold=0.5):
        y_hat = self.predict_prob(X)
        threshold_func = np.vectorize(lambda x: 1 if x > threshold else 0)
        y_pred = threshold_func(y_hat)
        return y_pred

    def score(self, X, y, threshold=0.5):
        y_predict = self.predict(X, threshold)
        return len(y_predict[y_predict == y]) / len(y_predict)
