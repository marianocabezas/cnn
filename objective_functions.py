from theano.tensor import sum, mean, std, log
import numpy as np


def probabilistic_dsc_objective(predictions, targets):
    multi = len(predictions.shape) > 1 and predictions.shape[1] > 1
    top = 2 * sum(predictions[:, 1] * targets) if multi else 2 * sum(predictions * targets)
    bottom = sum(predictions[:, 1]) + sum(targets) if multi else sum(predictions) + sum(targets)
    return 1-(top / bottom)


def cross_correlation(x, y):
    x_mean = mean(x)
    y_mean = mean(y)
    x_stdev = std(x)
    y_stdev = std(y)
    y_dev = y - y_mean
    x_dev = x - x_mean
    return 1 - (mean(x_dev*y_dev / (x_stdev*y_stdev)))


def logarithmic_dsc_objective(predictions, targets):
    top = log(2) + log(sum(predictions[:, 1] * targets))
    bottom = log(sum(predictions[:, 1]) + sum(targets))
    return -(top - bottom)


def accuracy_dsc_probabilistic(target, estimated):
    multi = len(estimated.shape) > 1 and estimated.shape[1] > 1
    top = 2 * np.sum(target * estimated[:, 1]) if multi else 2 * np.sum(target * estimated)
    bottom = np.sum(target) + np.sum(estimated[:, 1]) if multi else np.sum(target) + np.sum(estimated)
    return top / bottom


def accuracy_dsc(target, estimated):
    multi = len(estimated.shape) > 1 and estimated.shape[1] > 1
    a = target.astype(dtype=np.bool)
    b = np.array(estimated[:, 1] > 0.8) if multi else np.array(estimated > 0.8)
    return 2 * np.sum(np.logical_and(a, b)) / np.sum(np.sum(a) + np.sum(b))