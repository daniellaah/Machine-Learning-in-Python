# Reference: 李航. 《统计学习方法》 第2章 感知机
import numpy as np
import code
class Perceptron(object):
    '''
        X.ndim == 2, X.shape == (n_samples, n_features)
        y.ndim == 1, y.shape == (n_samples, )
        W.ndim == 2, W.shape == (n_features, 1)
        b is scalar
    '''
    def __init__(self, learning_rate=0.03, max_iter=100):
        self.weight = None # w
        self.bias = None # b
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        assert (X.shape[0] == y.shape[0])
        n_samples, n_features = X.shape
        self.weight = np.zeros((n_features, 1))
        self.bias = 0
        sample_idx = np.array([i for i in range(n_samples)])
        # for mis-classfied samples, y(w*x+b) <= 0
        mis_classfied_idx = sample_idx[(np.dot(X, self.weight) + self.bias).reshape(-1, ) * y <= 0]
        iteration = 0
        while(len(mis_classfied_idx) > 0 and iteration < self.max_iter):
            rand = np.random.choice(mis_classfied_idx)
            self.weight += (self.learning_rate * y[rand] * X[rand]).reshape(n_features, 1)
            self.bias += self.learning_rate * y[rand]
            mis_classfied_idx = sample_idx[(np.dot(X, self.weight) + self.bias).reshape(-1, ) * y <= 0]
            iteration += 1
        return self

    def predict(self, X):
        y_hat = np.dot(X, self.weight) + self.bias
        y_pred = np.apply_along_axis(lambda x: 1 if x >= 0 else -1, 1, y_hat)
        return y_pred

    def score(self, X, y):
        assert (X.shape[0] == y.shape[0])
        y_predict = self.predict(X)
        return len(y_predict[y_predict == y]) / len(y_predict)
