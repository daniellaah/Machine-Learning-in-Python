import numpy as np
from mlinpy.utils.metrics import accuracy_score
from mlinpy.utils.functions import gaussian
class BernoulliNaiveBayes():
    def __init__(self):
        self.prior_prob = {}
        self.condition_prob = {}

    def fit(self, X, y, alpha=1):
        '''计算概率P(xi=1|y=k), p(y=k)
        Args:
            X: 训练样本, m * n
            y: 标签, n * 1
            alpha: 平滑参数
        '''
        X = np.where(X > 2, 1, 0)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        for k in self.classes:
            n_samples_k = len(y[y == k])
            # 计算log prior probability log(p(y))
            self.prior_prob[k] = np.log((n_samples_k + alpha) / (n_samples + n_classes * alpha))
            # 计算log conditiional probability log(p(x=1|y))
            self.condition_prob[k] = np.log((np.sum(X[y == k, :], axis=0) + alpha) / (n_samples_k + 2 * alpha))
        return self

    def __predict(self, sample):
        '''对单个样本预测, 将概率连乘改为log概率连加, 防止浮点数下溢
        Args:
            sample: e.g. [1, 0, 0, 1 ...] feature_nums * 1
        '''
        result = {}
        for k in self.classes:
            log_post_prob = np.sum(self.condition_prob[k][sample == 1]) + \
                            np.sum(1- self.condition_prob[k][sample == 0])
            log_prior_prob = self.prior_prob[k]
            result[k] = log_post_prob + log_prior_prob
        return max(result, key=result.get)

    def predict(self, X):
        '''给定样本进行预测
        Args:
            X: 所有样本或者单个样本
        '''
        X = np.where(X > 2, 1, 0)
        if X.ndim == 1:
            return self._predict(X)
        else:
            return np.apply_along_axis(self.__predict, 1, X)

    def score(self, X, y_true):
        '''给定测试样本, 进行评估
        Args:
            X: 测试集
            y_true: 测试集标签
        '''
        y_predict = self.predict(X)
        return accuracy_score(y_predict, y_true)

class MultinomialNaiveBayes():
    def __init__(self):
        self.prior_prob = {}
        self.condition_prob = {}

    def fit(self, X, y, alpha=1):
        '''计算概率P(xi=1|y=k), p(y=k)
        Args:
            X: 训练样本, m * n
            y: 标签, n * 1
            alpha: 平滑参数
        '''
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        for k in self.classes:
            n_samples_k = len(y[y == k])
            # 计算log prior probability log(p(y))
            self.prior_prob[k] = np.log((n_samples_k + alpha) / (n_samples + n_classes * alpha))
            # 计算log conditiional probability log(p(x=1|y))
            self.condition_prob[k] = np.log((np.sum(X[y == k, :], axis=0) + alpha) / (n_samples_k + 2 * alpha))
        return self

    def __predict(self, sample):
        '''对单个样本预测, 将概率连乘改为log概率连加, 防止浮点数下溢
        Args:
            sample: e.g. [1, 0, 0, 1 ...] feature_nums * 1
        '''
        result = {}
        for k in self.classes:
            log_post_prob = np.sum(self.condition_prob[k] * sample) + \
                            np.sum(1 - self.condition_prob[k][sample == 0])
            log_prior_prob = self.prior_prob[k]
            result[k] = log_post_prob + log_prior_prob
        return max(result, key=result.get)

    def predict(self, X):
        '''给定样本进行预测
        Args:
            X: 所有样本或者单个样本
        '''
        if X.ndim == 1:
            return self._predict(X)
        else:
            return np.apply_along_axis(self.__predict, 1, X)

    def score(self, X, y_true):
        '''给定测试样本, 进行评估
        Args:
            X: 测试集
            y_true: 测试集标签
        '''
        y_predict = self.predict(X)
        return accuracy_score(y_predict, y_true)

class GuassianNaiveBayes():
    def __init__(self):
        self.prior_prob = {}
        self.condition_prob = {}

    def fit(self, X, y):
        '''
        Args:
            X: 训练样本, m * n
            y: 标签, n * 1
            alpha: 平滑参数
        '''
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        for k in self.classes:
            n_samples_k = len(y[y == k])
            # 计算prior probability log(p(y))
            self.prior_prob[k] = n_samples_k / n_samples
            # 计算conditiional probability log(p(x=1|y))
            mu, sigma = np.mean(X[y == k, :], axis=0), np.std(X[y == k, :], axis=0)
            self.condition_prob[k] = (mu, sigma)
        return self

    def __predict(self, sample):
        '''对单个样本预测, 将概率连乘改为log概率连加, 防止浮点数下溢
        Args:
            sample: e.g. [1, 0, 0, 1 ...] feature_nums * 1
        '''
        result = {}
        for k in self.classes:
            mu, sigma = self.condition_prob[k]
            log_post_prob = np.sum(np.log(gaussian(mu, sigma, sample)))
            log_prior_prob = np.log(self.prior_prob[k])
            result[k] = log_post_prob + log_prior_prob
        import code
        code.interact(local=locals())
        return max(result, key=result.get)

    def predict(self, X):
        '''给定样本进行预测
        Args:
            X: 所有样本或者单个样本
        '''
        if X.ndim == 1:
            return self._predict(X)
        else:
            return np.apply_along_axis(self.__predict, 1, X)

    def score(self, X, y_true):
        '''给定测试样本, 进行评估
        Args:
            X: 测试集
            y_true: 测试集标签
        '''
        y_predict = self.predict(X)
        return accuracy_score(y_predict, np.array(y_true))
