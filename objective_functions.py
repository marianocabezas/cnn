from theano import tensor
import numpy as np


def probabilistic_dsc_objective(predictions, targets):
    top = 2 * tensor.sum(predictions[:, 1] * targets)
    bottom = tensor.sum(predictions[:, 1]) + tensor.sum(targets)
    return 1-(top / bottom)


def cross_correlation(x, y):
    x_mean = tensor.mean(x)
    y_mean = tensor.mean(y)
    x_stdev = tensor.std(x)
    y_stdev = tensor.std(y)
    y_dev = y - y_mean
    x_dev = x - x_mean
    return 1 - (tensor.sum(x_dev*y_dev / (x_stdev*y_stdev)))


def logarithmic_dsc_objective(predictions, targets):
    top = tensor.log(2) + tensor.log(tensor.sum(predictions[:, 1] * targets))
    bottom = tensor.log(tensor.sum(predictions[:, 1]) + tensor.sum(targets))
    return -(top - bottom)


def accuracy_dsc_probabilistic(target, estimated):
    return 2 * np.sum(target * estimated[:, 1]) / (np.sum(target) + np.sum(estimated[:, 1]))


def accuracy_dsc(target, estimated):
    a = target.astype(dtype=np.bool)
    b = np.array(estimated[:, 1] > 0.8).astype(dtype=np.bool)
    return 2 * np.sum(np.logical_and(a, b)) / np.sum(np.sum(a) + np.sum(b))