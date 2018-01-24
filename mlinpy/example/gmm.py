import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.utils import shuffle
from mlinpy.models.gmm import GMM


if __name__ == '__main__':
    mu1, sigma1 = [0, 0], [[1, 0], [0, 1]]
    mu2, sigma2 = [2, 4], [[3, 0], [0, 1]]
    np.random.seed(8827)
    X1 = np.random.multivariate_normal(mu1, sigma1, 500)
    X2 = np.random.multivariate_normal(mu2, sigma2, 300)
    y = np.array([1]*500 + [0]*300)
    X = np.vstack([X1, X2])
    X, y = shuffle(X, y)
    gmm = GMM(n_components=2).fit(X)
    weight1, weight2 = gmm.weights_
    predict_mu1, predict_mu2 = gmm.means_
    predict_sigma1, predict_sigma2 = gmm.covariances_
    predict_gauss1 = multivariate_normal(predict_mu1, predict_sigma1)
    predict_gauss2 = multivariate_normal(predict_mu2, predict_sigma2)
    predict_y1 = predict_gauss1.pdf(X)
    predict_y2 = predict_gauss2.pdf(X)
    predict1 = (predict_y1 > predict_y2).astype(int)
    predict2 = (predict_y1 < predict_y2).astype(int)
    acc1, acc2 = np.mean(predict1 == y), np.mean(predict2 == y)
    print('accuracy: {}'.format(acc1 if acc1 > acc2 else acc2))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax.set_title('True')
    ax.scatter(X[predict1==1, 0], X[predict1==1, 1], c='r', s=10)
    ax.scatter(X[predict1==0, 0], X[predict1==0, 1], c='b', s=10)
    ax = fig.add_subplot(122)
    ax.set_title('Predict')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='r', s=10)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='b', s=10)
    plt.show()
