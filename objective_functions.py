from theano import tensor
import numpy as np


def probabilistic_dsc_objective(predictions, targets):
    top = 2 * tensor.sum(predictions[:, 1] * targets)
    bottom = tensor.sum(predictions[:, 1]) + tensor.sum(targets)
    return -(top / bottom)


def logarithmic_dsc_objective(predictions, targets):
    top = tensor.log(2) + tensor.log(tensor.sum(predictions[:, 1] * targets))
    bottom = tensor.log(tensor.sum(predictions[:, 1]) + tensor.sum(targets))
    return -(top - bottom)


def accuracy_dsc_probabilistic(target, estimated):
    return 2 * np.sum(target * estimated[:, 1]) / (np.sum(target) + np.sum(estimated[:, 1]))


def accuracy_dsc(target, estimated):
    A = target.astype(dtype=np.bool)
    B = np.array(estimated[:, 1] > 0.8).astype(dtype=np.bool)
    return 2 * np.sum(np.logical_and(A, B)) / np.sum(np.sum(A) + np.sum(B))