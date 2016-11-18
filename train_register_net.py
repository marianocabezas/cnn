from __future__ import print_function
import argparse
import os
import sys
from time import strftime
import numpy as np
from nets import create_cnn3d_register
from utils import color_codes
from data_creation import load_patch_batch_percent
from data_creation import load_iter1_data, load_iter2_data
from nibabel import load as load_nii
from data_manipulation.metrics import dsc_seg, tp_fraction_seg, fp_fraction_seg


def parse_inputs():
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Subtraction/')
    parser.add_argument('-l', '--pool-size', dest='pool_size', type=int, default=2)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-n', '--num-filters', action='store', dest='number_filters', nargs='+', type=int, default=32)
    parser.add_argument('-i', '--input', action='store', dest='input_size', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('--baseline-folder', action='store', dest='b_folder', default='time1/')
    parser.add_argument('--followup-folder', action='store', dest='f_folder', default='time2/')
    parser.add_argument('--padding', action='store', dest='padding', default='valid')
    return vars(parser.parse_args())


def main():
    c = color_codes()
    options = parse_inputs()

    dir_name = options['dir_name']
    patients = [f for f in sorted(os.listdir(dir_name))
                if os.path.isdir(os.path.join(dir_name, f))]
    n_patients = len(patients)
    names = get_names_from_path(dir_name, options, patients)

    input_size = options['input_size']

    seed = np.random.randint(np.iinfo(np.int32).max)

    metrics_file = os.path.join(dir_name, 'metrics' + sufix)

    with open(metrics_file, 'w') as f:
        print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting leave-one-out' + c['nc'])

        for i in range(0, n_patients):
            case = patients[i]
            path = os.path.join(dir_name, case)
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Patient ' + c['b'] + case + c['nc'] +
                  c['g'] + ' (%d/%d)' % (i + 1, n_patients))
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  '<Running iteration ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
            net_name = os.path.join(path, 'deep-exp_longitudinal.init' + sufix + '.')
            net = create_cnn3d_register(
                input_shape=(None, names.shape[0], input_size),
                patience=10,
                name=net_name,
                epochs=200
            )