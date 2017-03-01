import numpy as np
class Perceptron(object):
    def __init__(self, alpha=0.03, w=[0.0,0.0,0.0]):
        self.alpha = alpha
        # 权值
        self.w = np.array(w, dtype=float)
        self.w.shape = (3,1)

    def has_misclassified(self, X, y):
        # 是否存在误分类点
        # 若不存在, 返回 False, None, None
        # 若存在, 返回 True, 第一个误分类点, 第一个误分类点的类别
        y_pre = y*(np.dot(X, self.w))
        for i in range(X.shape[0]):
            if y_pre[i] <= 0:
                x_error = X[i]
                x_error.shape = (3,1)
                return True, x_error, y[i]
        return False, None, None

    def update_w(self, x, y):
        # 根据传入的误分类点更新权值
        self.w += self.alpha * y * x

    def fit(self, X, y):
        # 拟合样本, 注意, 样本必须是线性可分, 否则死循环
        y.shape = (X.shape[0], 1)
        while True:
            has_mis, x_mis, y_mis = self.has_misclassified(X, y)
            if not has_mis:
                break
            else:
                self.update_w(x_mis, y_mis)

    def predict(self, x):
        # 对一个样本进行预测, 返回1或0
        bias_unit = np.ones((1,1))
        x = np.concatenate((bias_unit, x), axis=0)
        return 1 if np.dot(self.w.T, x) > 0 else 0

if __name__ == '__main__':
    model = Perceptron(alpha=1)
    X = np.array([[1,3,3], [1,4,3], [1,1,1]])
    y = np.array([1,1,-1])
    model.fit(X, y)

    x = np.array([10,10])
    x.shape = (2,1)
    print(model.predict(x))
