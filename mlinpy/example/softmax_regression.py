from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlinpy.models.softmax_regression import SoftmaxRegression

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8827)
    clf = SoftmaxRegression([0, 1, 2], 0.01, 0.01, 1000).fit(X_train, y_train, True, 100)
    print('Accuracy: {:.2f}%'.format(clf.score(X_test, y_test) * 100))
