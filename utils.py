import numpy as np
from math import floor


def color_codes():
    codes = {'g': '\033[32m',
             'c': '\033[36m',
             'bg': '\033[32;1m',
             'b': '\033[1m',
             'nc': '\033[0m',
             'gc': '\033[32m;0m',
             'lgy': '\033[37m',
             'dgy': '\033[30m',
             }
    return codes


def random_affine3d_matrix(x_range=np.pi, y_range=np.pi, z_range=np.pi, t_range=5):
    x_angle = x_range * np.random.random() - (x_range / 2)
    y_angle = y_range * np.random.random() - (y_range / 2)
    z_angle = z_range * np.random.random() - (z_range / 2)
    t = t_range * np.random.random(3) - (t_range / 2)

    sx = np.sin(x_angle)
    cx = np.cos(x_angle)
    sy = np.sin(y_angle)
    cy = np.cos(y_angle)
    sz = np.sin(z_angle)
    cz = np.cos(z_angle)

    affine = np.array([
        [cy*cz, sx*sy*cz+cx*sz, -cx*sy*cz+sx*sz, t[0]],
        [-cy*sz, -sx*sy*sx+cx*cz, cx*sy*sz+sx*cz, t[1]],
        [sy, -sx*cy, cx*cy, t[2]],
        [0, 0, 0, 1],
    ])

    return affine


def train_test_split(data, labels, test_size=0.1, random_state=42):
    # Init (Set the random seed and determine the number of cases for test)
    n_test = int(floor(data.shape[0]*test_size))

    # We create a random permutation of the data
    # First we permute the data indices, then we shuffle the data and labels
    np.random.seed(random_state)
    indices = np.random.permutation(range(0, data.shape[0])).tolist()
    np.random.seed(random_state)
    shuffled_data = np.random.permutation(data)
    np.random.seed(random_state)
    shuffled_labels = np.random.permutation(labels)

    x_train = shuffled_data[:-n_test]
    x_test = shuffled_data[-n_test:]
    y_train = shuffled_labels[:-n_test]
    y_test = shuffled_data[-n_test:]
    idx_train = indices[:-n_test]
    idx_test = indices[-n_test:]

    return x_train, x_test, y_train, y_test, idx_train, idx_test


def leave_one_out(data_list, labels_list):
    for i in range(0, len(data_list)):
        yield data_list[:i] + data_list[i+1:], labels_list[:i] + labels_list[i+1:], i
