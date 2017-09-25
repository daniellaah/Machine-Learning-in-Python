from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlinpy.models.naive_bayes import GuassianNaiveBayes

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    import code
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8827)
    clf = GuassianNaiveBayes().fit(X_train, y_train)
    print('准确率: {:2f}%'.format(clf.score(X_test, y_test) * 100))
