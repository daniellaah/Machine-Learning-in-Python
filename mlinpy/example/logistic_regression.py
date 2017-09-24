from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlinpy.models.logistic_regression import LogisticRegression

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # 该数据集一共有3类, 我们只用y=0和y=1这两个类, 并且将标签y=0改为y=-1
    X = X[y < 2]
    y = y[y < 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8827)
    clf = LogisticRegression(0.03, 100).fit(X_train, y_train, True, 10)
    print('准确率: {}%'.format(clf.score(X_test, y_test) * 100))
