import numpy as np
from lasagne.layers import InputLayer
import re
from math import floor
import pickle


def color_codes():
    codes = {
        'nc': '\033[0m',
        'b': '\033[1m',
        'k': '\033[0m',
        '0.25': '\033[30m',
        'dgy': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'gc': '\033[32m;0m',
        'bg': '\033[32;1m',
        'y': '\033[33m',
        'c': '\033[36m',
        '0.75': '\033[37m',
        'lgy': '\033[37m',
    }
    return codes


def inverse_color_codes():
    codes = {
        '\033[30m': '0.25',
        '\033[31m': 'r',
        '\033[32m': 'g',
        '\033[33m': 'y',
        '\033[34m': 'b',
        '\033[35m': 'm',
        '\033[36m': 'c',
        '\033[37m': '0.75',
    }
    return codes


def name_and_color(layer_name):
    color_match = re.search('\x1b\[([0-9]+;*[0-9]*)m', layer_name) if layer_name else None
    color = inverse_color_codes()[color_match.group(0)] if color_match else 'k'
    name = ''.join(re.split('\x1b\[[0-9]+;*[0-9]*m', layer_name)) if layer_name else ''

    return color, name


def get_layer_depth(layer):
    if isinstance(layer, InputLayer):
        return 0
    else:
        try:
            inputs = layer.input_layers
        except AttributeError:
            inputs = [layer.input_layer]
    return max([get_layer_depth(l) for l in inputs]) + 1


def plot_layer(layer, xmin, xmax, ymax, height, fontsize, fig):
    color, name = name_and_color(layer.name)
    c = color_codes()
    print(c[color] + '%s ([%f-%f]: %f)' % (name, xmin, xmax, (xmax + xmin) / 2.0) + c['nc'])
    fig.text(
        (xmax + xmin)/2.0,
        (ymax - get_layer_depth(layer) + 1) / height,
        name,
        ha='center',
        va='center',
        color=color,
        size=fontsize,
        transform=fig.transFigure,
        bbox=dict(boxstyle='round', fc='w', ec='k')
    )

    if not isinstance(layer, InputLayer):
        try:
            inputs = layer.input_layers
        except AttributeError:
            inputs = [layer.input_layer]

        inputs_width = float((xmax - xmin))/len(inputs)
        inputs_xmin = [x * inputs_width for x in range(len(inputs))]
        inputs_xmax = [(x + 1.0) * inputs_width for x in range(len(inputs))]
        for xmin_i, xmax_i, i in zip(inputs_xmin, inputs_xmax, inputs):
            plot_layer(i, xmin + xmin_i, xmin + xmax_i, ymax, height, fontsize, fig)


def plot_layer_tree(final_layer):
    import matplotlib
    matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt
    depth = get_layer_depth(final_layer)
    spacing = 1.2
    height = (spacing * (depth + 1) + .5)
    fontsize = 0.3 * 72

    fig = plt.figure(1, ((3 * height / 4) / 1.5, height / 1.5))

    plot_layer(final_layer, 0, 1, depth, height, fontsize, fig)

    plt.draw()
    plt.show()


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


class EarlyStopping(object):
    """From https://github.com/dnouri/kfkd-tutorial"""
    def __init__(self, patience=50):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_train = train_history[-1]['train_loss']
        current_epoch = train_history[-1]['epoch']

        # Ignore if training loss is greater than valid loss
        if current_train > current_valid:
            return

        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience <= current_epoch:
            print('Early stopping.')
            print('Best valid loss was {:.6f} at epoch {}.'.format(
                self.best_valid, self.best_valid_epoch))
            nn.get_all_params_values()
            raise StopIteration()


class WeightsLogger(object):
    """From https://github.com/dnouri/kfkd-tutorial"""
    def __init__(self, filename):
        self.params = []
        self.filename = filename

    def __call__(self, nn, train_history):
        params = dict(
            train=train_history[-1]['train_loss'],
            valids=train_history[-1]['valid_loss'],
            weights=nn.get_all_params()
        )
        self.params.append(params)

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.params, f, -1)
