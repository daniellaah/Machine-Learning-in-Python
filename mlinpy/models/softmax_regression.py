# Reference: http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
import numpy as np
from mlinpy.utils.functions import softmax

import code
class SoftmaxRegression():
    def __init__(self, classes, learning_rate=0.01, lambd=0.01, max_iter=200):
        self.n_classes = len(classes)
        self.theta = None # (n_classes, n_features)
        self.index_to_label = {}
        self.label_to_index = {}
        for i, label in enumerate(classes):
            self.index_to_label[i] = label
            self.label_to_index[label] = i
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.max_iter = max_iter

    def __comput_cost(self, biased_X, Y):
        Y_prob = softmax(self.theta, biased_X)
        cost = -np.sum(Y * np.log(Y_prob)) / biased_X.shape[0]
        reg = np.sum(np.square(self.theta)) * self.lambd / 2
        return cost + reg

    def fit(self, X, y, print_cost=True, print_num=100):
        n_samples = X.shape[0]
        biased_X = np.column_stack((np.ones(n_samples), X))
        n_features = biased_X.shape[1]
        Y = np.zeros((self.n_classes, n_samples))
        for i, label in enumerate(y):
            Y[self.label_to_index[label], i] = 1
        self.theta = np.zeros((self.n_classes, n_features), dtype=float)
        for i in range(self.max_iter):
            grad = -np.dot((Y - softmax(self.theta, biased_X)), biased_X)  / n_samples + self.lambd * self.theta
            self.theta -= self.learning_rate * grad
            cost = self.__comput_cost(biased_X, Y)
            if i % print_num == 0 and print_cost:
                print ("Cost after iteration {}: {:.6f}".format(i, cost))
        return self

    def predict_prob(self, X):
        '''
        Return:
            (n_classes, n_samples) probabilty matrix
        '''
        biased_X = np.column_stack((np.ones(X.shape[0]), X))
        return softmax(self.theta, biased_X)

    def predict(self, X):
        y_prob = self.predict_prob(X)
        index_result = np.argmax(y_prob, axis=0)
        return np.array([self.index_to_label[index] for index in index_result])

    def score(self, X, y):
        y_pred = self.predict(X)
        return len(y_pred[y_pred == y]) / len(y_pred)
