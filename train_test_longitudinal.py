from __future__ import print_function
import argparse
import os
import sys
from time import strftime
import numpy as np
from nets import create_cnn3d_det_string
from data_creation import load_patch_batch_percent
from data_creation import load_iter1_data, load_iter2_data
from nibabel import load as load_nii
from data_manipulation.metrics import dsc_seg, tp_fraction_seg, fp_fraction_seg


def color_codes():
    codes = {'g': '\033[32m',
             'c': '\033[36m',
             'bg': '\033[32;1m',
             'b': '\033[1m',
             'nc': '\033[0m',
             'gc': '\033[32m, \033[0m'
             }
    return codes


def main():

    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Subtraction/')
    parser.add_argument('-p', '--patch-width', dest='patch_width', type=int, default=9)
    parser.add_argument('-c', '--conv-width', dest='conv_width', type=int, default=3)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10000)
    parser.add_argument('-l', '--layers', action='store', dest='layers', default='ca')
    parser.add_argument('-n', '--number-filters', action='store', dest='number_filters', type=int, default=32)
    parser.add_argument('--prefix-folder', dest='prefix', default='time2/preprocessed/')
    parser.add_argument('--flair-baseline', action='store', dest='flair_b', default='flair_moved.nii.gz')
    parser.add_argument('--pd-baseline', action='store', dest='pd_b', default='pd_moved.nii.gz')
    parser.add_argument('--t2-baseline', action='store', dest='t2_b', default='t2_moved.nii.gz')
    parser.add_argument('--flair-12m', action='store', dest='flair_f', default='flair_registered.nii.gz')
    parser.add_argument('--pd-12m', action='store', dest='pd_f', default='pd_corrected.nii.gz')
    parser.add_argument('--t2-12m', action='store', dest='t2_f', default='t2_corrected.nii.gz')
    parser.add_argument('--mask', action='store', dest='mask', default='gt_mask.nii')
    parser.add_argument('--wm-mask', action='store', dest='wm_mask', default='union_wm_mask.nii.gz')
    parser.add_argument('--padding', action='store', dest='padding', default='valid')
    options = vars(parser.parse_args())

    c = color_codes()
    padding = options['padding']
    layers = ''.join(options['layers'])
    n_filters = options['number_filters']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    conv_width = options['conv_width']
    sufix = '.%s.p%d.c%d.n%d.pad_%s' % (layers, patch_width, conv_width, n_filters, padding)
    # Create the data
    prefix_name = options['prefix']
    flair_b_name = os.path.join(prefix_name, options['flair_b'])
    pd_b_name = os.path.join(prefix_name, options['pd_b'])
    t2_b_name = os.path.join(prefix_name, options['t2_b'])
    flair_f_name = os.path.join(prefix_name, options['flair_f'])
    pd_f_name = os.path.join(prefix_name, options['pd_f'])
    t2_f_name = os.path.join(prefix_name, options['t2_f'])
    mask_name = options['mask']
    wm_name = options['wm_mask']
    dir_name = options['dir_name']
    patients = [f for f in sorted(os.listdir(dir_name))
                if os.path.isdir(os.path.join(dir_name, f))]
    n_patients = len(patients)
    flair_b_names = [os.path.join(dir_name, patient, flair_b_name) for patient in patients]
    pd_b_names = [os.path.join(dir_name, patient, pd_b_name) for patient in patients]
    t2_b_names = [os.path.join(dir_name, patient, t2_b_name) for patient in patients]
    flair_f_names = [os.path.join(dir_name, patient, flair_f_name) for patient in patients]
    pd_f_names = [os.path.join(dir_name, patient, pd_f_name) for patient in patients]
    t2_f_names = [os.path.join(dir_name, patient, t2_f_name) for patient in patients]
    names = np.stack([name for name in [flair_f_names, pd_f_names, t2_f_names, flair_b_names, pd_b_names, t2_b_names]])
    channels = names.shape[0]
    seed = np.random.randint(np.iinfo(np.int32).max)

    metrics_file = os.path.join(dir_name, 'metrics' + sufix)
    with open(metrics_file, 'w') as f:

        print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting leave-one-out' + c['nc'])

        for i in range(0, n_patients):
            case = patients[i]
            path = os.path.join(dir_name, case)
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Patient ' + c['b'] + case + c['nc'] +
                  c['g'] + ' (%d/%d)' % (i+1, n_patients))
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  '<Running iteration ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
            net_name = os.path.join(path, 'deep-longitudinal.init' + sufix + '.')
            net = create_cnn3d_det_string(
                layers,
                (None, channels, patch_width, patch_width, patch_width),
                conv_width,
                padding,
                2,
                n_filters,
                10,
                True,
                net_name,
                200
            )
            flair_b_test = os.path.join(path, flair_b_name)
            pd_b_test = os.path.join(path, pd_b_name)
            t2_b_test = os.path.join(path, t2_b_name)
            flair_f_test = os.path.join(path, flair_f_name)
            pd_f_test = os.path.join(path, pd_f_name)
            t2_f_test = os.path.join(path, t2_f_name)
            names_test = np.array([flair_f_test, pd_f_test, t2_f_test, flair_b_test, pd_b_test, t2_b_test])
            outputname1 = os.path.join(path, 'test' + str(i) + sufix + '.iter1.nii.gz')
            try:
                net.load_params_from(net_name + 'model_weights.pkl')
            except IOError:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                      c['g'] + 'Loading the data for ' + c['b'] + 'iteration 1' + c['nc'])
                names_lou = np.concatenate([names[:, :i], names[:, i + 1:]], axis=1)
                paths = [os.path.join(dir_name, p) for p in np.concatenate([patients[:i], patients[i+1:]])]
                mask_names = [os.path.join(p_path, mask_name) for p_path in paths]
                wm_names = [os.path.join(p_path, wm_name) for p_path in paths]

                x_train, y_train = load_iter1_data(
                    names_lou=names_lou,
                    mask_names=mask_names,
                    roi_names=wm_names,
                    patch_size=patch_size,
                    seed=seed
                )

                print('                Training vector shape ='
                      ' (' + ','.join([str(length) for length in x_train.shape]) + ')')
                print('                Training labels shape ='
                      ' (' + ','.join([str(length) for length in y_train.shape]) + ')')

                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                      'Training (' + c['b'] + 'initial' + c['nc'] + c['g'] + ')' + c['nc'])
                # We try to get the last weights to keep improving the net over and over
                net.fit(x_train, y_train)

            try:
                image_nii = load_nii(outputname1)
                image1 = image_nii.get_data()
            except IOError:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                      '<Creating the probability map ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
                image_nii = load_nii(os.path.join(path, flair_f_name))
                image1 = np.zeros_like(image_nii.get_data())
                print('              0% of data tested', end='\r')
                sys.stdout.flush()
                for batch, centers, percent in load_patch_batch_percent(names_test, batch_size, patch_size):
                    y_pred = net.predict_proba(batch)
                    print('              %f%% of data tested' % percent, end='\r')
                    sys.stdout.flush()
                    [x, y, z] = np.stack(centers, axis=1)
                    image1[x, y, z] = y_pred[:, 1]

                image_nii.get_data()[:] = image1
                image_nii.to_filename(outputname1)

            ''' Here we get the seeds '''
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + '<Looking for seeds for the final iteration>' + c['nc'])
            for patient in np.rollaxis(np.concatenate([names[:, :i], names[:, i+1:]], axis=1), 1):
                patient_path = '/'.join(patient[0].rsplit('/')[:-1])
                outputname = os.path.join(patient_path, 'test' + str(i) + sufix + '.iter1.nii.gz')
                try:
                    load_nii(outputname)
                    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                          c['g'] + '     Patient ' + patient[0].rsplit('/')[-4] + ' already done' + c['nc'])
                except IOError:
                    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                          c['g'] + '     Testing with patient ' + c['b'] + patient[0].rsplit('/')[-4] + c['nc'])
                    image_nii = load_nii(patient[0])
                    image = np.zeros_like(image_nii.get_data())
                    print('                   0% of data tested', end='\r')
                    sys.stdout.flush()
                    for batch, centers, percent in load_patch_batch_percent(patient, 100000, patch_size):
                        y_pred = net.predict_proba(batch)
                        print('                   %f%% of data tested' % percent, end='\r')
                        sys.stdout.flush()
                        [x, y, z] = np.stack(centers, axis=1)
                        image[x, y, z] = y_pred[:, 1]

                    print(c['g'] + '                   -- Saving image ' + c['b'] + outputname + c['nc'])
                    image_nii.get_data()[:] = image
                    image_nii.to_filename(outputname)

            ''' Here we perform the last iteration '''
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  '<Running iteration ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
            outputname2 = os.path.join(path, 'test' + str(i) + sufix + '.iter2.nii.gz')
            net_name = os.path.join(path, 'deep-longitudinal.final' + sufix + '.')
            net = create_cnn3d_det_string(
                layers,
                (None, channels, patch_width, patch_width, patch_width),
                conv_width,
                padding,
                2,
                n_filters,
                50,
                True,
                net_name,
                2000
            )

            try:
                net.load_params_from(net_name + 'model_weights.pkl')
            except IOError:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                      c['g'] + 'Loading the data for ' + c['b'] + 'iteration 2' + c['nc'])
                names_lou = np.concatenate([names[:, :i], names[:, i + 1:]], axis=1)
                roi_paths = ['/'.join(name.rsplit('/')[:-1]) for name in names_lou[0, :]]
                paths = [os.path.join(dir_name, p) for p in np.concatenate([patients[:i], patients[i + 1:]])]
                roi_names = [os.path.join(p_path, 'test' + str(i) + sufix + '.iter1.nii.gz') for p_path in roi_paths]
                mask_names = [os.path.join(p_path, mask_name) for p_path in paths]

                x_train, y_train = load_iter2_data(
                    names_lou=names_lou,
                    mask_names=mask_names,
                    roi_names=roi_names,
                    patch_size=patch_size,
                    seed=seed
                )

                print('              Training vector = (' + ','.join([str(length) for length in x_train.shape]) + ')')
                print('              Training labels = (' + ','.join([str(length) for length in y_train.shape]) + ')')
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                      c['g'] + 'Training (' + c['b'] + 'final' + c['nc'] + c['g'] + ')' + c['nc'])
                net.fit(x_train, y_train)
            try:
                image_nii = load_nii(outputname2)
                image2 = image_nii.get_data()
            except IOError:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                      '<Creating the probability map ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
                image_nii = load_nii(os.path.join(path, flair_f_name))
                image2 = np.zeros_like(image_nii.get_data())
                print('              0% of data tested', end='\r')
                sys.stdout.flush()
                for batch, centers, percent in load_patch_batch_percent(names_test, batch_size, patch_size):
                    y_pred = net.predict_proba(batch)
                    print('              %f%% of data tested' % percent, end='\r')
                    sys.stdout.flush()
                    [x, y, z] = np.stack(centers, axis=1)
                    image2[x, y, z] = y_pred[:, 1]

                image_nii.get_data()[:] = image2
                image_nii.to_filename(outputname2)

            image = (image1 * image2) > 0.5
            image_nii.get_data()[:] = image
            outputname_final = os.path.join(path, 'test' + str(i) + sufix + '.final.nii.gz')
            image_nii.to_filename(outputname_final)

            gt = load_nii(os.path.join(path, mask_name)).get_data().astype(dtype=np.bool)
            seg1 = image1 > 0.5
            seg2 = image1 > 0.5
            dsc1 = dsc_seg(gt, seg1)
            dsc2 = dsc_seg(gt, seg2)
            dsc_final = dsc_seg(gt, image)
            tpf1 = tp_fraction_seg(gt, seg1)
            tpf2 = tp_fraction_seg(gt, seg2)
            tpf_final = tp_fraction_seg(gt, image)
            fpf1 = fp_fraction_seg(gt, seg1)
            fpf2 = fp_fraction_seg(gt, seg2)
            fpf_final = fp_fraction_seg(gt, image)
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  '<DSC ' + c['c'] + case + c['g'] + ' = ' + c['b'] + str(dsc_final) + c['nc'] + c['g'] + '>' + c['nc'])
            f.write('%s;Test 1; %f;%f;%f\n' % (case, dsc1, tpf1, fpf1))
            f.write('%s;Test 2; %f;%f;%f\n' % (case, dsc2, tpf2, fpf2))
            f.write('%s;Final; %f;%f;%f\n' % (case, dsc_final, tpf_final, fpf_final))


if __name__ == '__main__':
    main()
