import numpy as np

def accuracy_score(y_true, y_pred):
    assert (y_true.shape == y_pred.shape)
    return (y_true[y_true == y_pred]).shape[0] / y_true.shape[0]
