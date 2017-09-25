import numpy as np

def sigmoid(z):
       return 1.0 / (1 + np.exp(-z))

def softmax(theta, x):
    '''
    Args:
        theta: theta.shape == (n_class, n_features)
        x: x.shape == (n_samples, n_features)
    Return:
        normalized probabilty matrix
    '''
    prob_matrix = np.exp(np.dot(theta, x.T))
    nomalized = prob_matrix / np.sum(prob_matrix, axis=0)
    return nomalized

def gaussian(mu, sigma, x):
    exponent = np.exp(- ((x - mu)**2 / (2 * sigma**2)))
    return np.log(exponent / (np.sqrt(2 * np.pi) * sigma))
