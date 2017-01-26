import numpy as np


def color_codes():
    codes = {'g': '\033[32m',
             'c': '\033[36m',
             'bg': '\033[32;1m',
             'b': '\033[1m',
             'nc': '\033[0m',
             'gc': '\033[32m, \033[0m'
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
