from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlinpy.models.naive_bayes import BernoulliNaiveBayes

if __name__ == '__main__':
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    import code
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8827)
    clf = BernoulliNaiveBayes().fit(X_train, y_train, 1)
    print('准确率: {}%'.format(clf.score(X_test, y_test) * 100))
