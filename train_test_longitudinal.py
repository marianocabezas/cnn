from __future__ import print_function
import argparse
import os
import sys
from time import strftime
import numpy as np
from data_creation import load_patch_batch_percent, load_thresholded_norm_images_by_name
from data_creation import load_patch_vectors_by_name_pr, load_patch_vectors_by_name, load_mask_vectors
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv3DDNNLayer, Pool3DDNNLayer
from lasagne import nonlinearities, objectives, updates
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.hooks import EarlyStopping
from nibabel import load as load_nii


def color_codes():
    codes = {'g': '\033[32m',
             'c': '\033[36m',
             'bg': '\033[32;1m',
             'b': '\033[1m',
             'nc': '\033[0m',
             'gc': '\033[32m, \033[0m'
             }
    return codes


def load_and_stack_iter1(names_lou, mask_names, patch_size):
    rois = load_thresholded_norm_images_by_name(names_lou[0, :], threshold=1.0)
    images_loaded = [load_patch_vectors_by_name(names_i, mask_names, patch_size, rois=rois)
                     for names_i in names_lou]

    x_train = [np.stack(images, axis=1) for images in zip(*images_loaded)]
    y_train = [np.concatenate([np.ones(x.shape[0]/2), np.zeros(x.shape[0]/2)]) for x in x_train]

    return x_train, y_train


def load_and_stack_iter2(names_lou, mask_names, roi_names, patch_size):
    pr_maps = [load_nii(roi_name).get_data() for roi_name in roi_names]
    images_loaded = [load_patch_vectors_by_name_pr(names_i, mask_names, patch_size, pr_maps=pr_maps)
                     for names_i in names_lou]

    x_train = [np.stack(images, axis=1) for images in zip(*images_loaded)]
    y_train = [np.concatenate([np.ones(x.shape[0]/2), np.zeros(x.shape[0]/2)]) for x in x_train]

    return x_train, y_train


def concatenate_and_permute(x, y, seed):
    print('                Creating data vector')
    x_train = np.concatenate(x)
    y_train = np.concatenate(y)

    print('                Permuting the data')
    np.random.seed(seed)
    x_train = np.random.permutation(x_train.astype(dtype=np.float32))
    print('                Permuting the labels')
    np.random.seed(seed)
    y_train = np.random.permutation(y_train.astype(dtype=np.int32))

    return x_train, y_train


def load_iter1_data(names_lou, mask_names, patch_size, seed):
    x_train, y_train = load_and_stack_iter1(names_lou, mask_names, patch_size)
    x_train, y_train = concatenate_and_permute(x_train, y_train, seed)

    return x_train, y_train


def load_iter2_data(names_lou, mask_names, roi_names, patch_size, seed):
    x_train, y_train = load_and_stack_iter2(names_lou, mask_names, roi_names, patch_size)
    x_train, y_train = concatenate_and_permute(x_train, y_train, seed)

    return x_train, y_train


def main():

    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Subtraction/')
    parser.add_argument('-p', '--prefix-folder', dest='prefix', default='time2/preprocessed/')
    parser.add_argument('--flair-baseline', action='store', dest='flair_b', default='flair_moved.nii.gz')
    parser.add_argument('--pd-baseline', action='store', dest='pd_b', default='pd_moved.nii.gz')
    parser.add_argument('--t2-baseline', action='store', dest='t2_b', default='t2_moved.nii.gz')
    parser.add_argument('--flair-12m', action='store', dest='flair_f', default='flair_registered.nii.gz')
    parser.add_argument('--pd-12m', action='store', dest='pd_f', default='pd_corrected.nii.gz')
    parser.add_argument('--t2-12m', action='store', dest='t2_f', default='t2_corrected.nii.gz')
    parser.add_argument('--mask', action='store', dest='mask', default='gt_mask.nii')
    options = vars(parser.parse_args())

    c = color_codes()
    patch_size = (11, 11, 11)
    batch_size = 100000
    # Create the data
    prefix_name = options['prefix']
    flair_b_name = os.path.join(prefix_name, options['flair_b'])
    pd_b_name = os.path.join(prefix_name, options['pd_b'])
    t2_b_name = os.path.join(prefix_name, options['t2_b'])
    flair_f_name = os.path.join(prefix_name, options['flair_f'])
    pd_f_name = os.path.join(prefix_name, options['pd_f'])
    t2_f_name = os.path.join(prefix_name, options['t2_f'])
    mask_name = options['mask']
    dir_name = options['dir_name']
    patients = [f for f in sorted(os.listdir(dir_name))
                if os.path.isdir(os.path.join(dir_name, f))]
    flair_b_names = [os.path.join(dir_name, patient, flair_b_name) for patient in patients]
    pd_b_names = [os.path.join(dir_name, patient, pd_b_name) for patient in patients]
    t2_b_names = [os.path.join(dir_name, patient, t2_b_name) for patient in patients]
    flair_f_names = [os.path.join(dir_name, patient, flair_f_name) for patient in patients]
    pd_f_names = [os.path.join(dir_name, patient, pd_f_name) for patient in patients]
    t2_f_names = [os.path.join(dir_name, patient, t2_f_name) for patient in patients]
    names = np.stack([name for name in [flair_f_names, pd_f_names, t2_f_names, flair_b_names, pd_b_names, t2_b_names]])
    seed = np.random.randint(np.iinfo(np.int32).max)

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting leave-one-out' + c['nc'])

    for i in range(0, 15):
        case = patients[i]
        path = os.path.join(dir_name, case)
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Patient ' + c['b'] + case + c['nc'])
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Running iteration ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
        net_name = os.path.join(path, 'deep-challenge2016.init.')
        net = NeuralNet(
            layers=[
                (InputLayer, dict(name='in', shape=(None, 4, 15, 15, 15))),
                (Conv3DDNNLayer, dict(name='conv1_1', num_filters=32, filter_size=(5, 5, 5), pad='same')),
                (Pool3DDNNLayer, dict(name='avgpool_1', pool_size=2, stride=2, mode='average_inc_pad')),
                (Conv3DDNNLayer, dict(name='conv2_1', num_filters=64, filter_size=(5, 5, 5), pad='same')),
                (Pool3DDNNLayer, dict(name='avgpool_2', pool_size=2, stride=2, mode='average_inc_pad')),
                (DropoutLayer, dict(name='l2drop', p=0.5)),
                (DenseLayer, dict(name='l1', num_units=256)),
                (DenseLayer, dict(name='out', num_units=2, nonlinearity=nonlinearities.softmax)),
            ],
            objective_loss_function=objectives.categorical_crossentropy,
            update=updates.adam,
            update_learning_rate=0.0001,
            on_epoch_finished=[
                SaveWeights(net_name + 'model_weights.pkl', only_best=True, pickle=False),
                EarlyStopping(patience=10)
            ],
            verbose=10,
            max_epochs=50,
            train_split=TrainSplit(eval_size=0.25),
            custom_scores=[('dsc', lambda pred, t: 2 * np.sum(pred * t[:, 1]) / np.sum((pred + t[:, 1])))],
        )
        flair_b_test = os.path.join(path, flair_b_name)
        pd_b_test = os.path.join(path, pd_b_name)
        t2_b_test = os.path.join(path, t2_b_name)
        flair_f_test = os.path.join(path, flair_f_name)
        pd_f_test = os.path.join(path, pd_f_name)
        t2_f_test = os.path.join(path, t2_f_name)
        names_test = np.array([flair_f_test, pd_f_test, t2_f_test, flair_b_test, pd_b_test, t2_b_test])
        outputname1 = os.path.join(path, 'test' + str(i) + '.iter1.nii.gz')
        try:
            net.load_params_from(net_name + 'model_weights.pkl')
        except IOError:
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Loading the data for ' + c['b'] + 'iteration 1' + c['nc'])
            names_lou = np.concatenate([names[:, :i], names[:, i + 1:]], axis=1)
            paths = [os.path.join(dir_name, p) for p in np.concatenate([patients[:i], patients[i+1:]])]
            mask_names = [os.path.join(p_path, mask_name) for p_path in paths]

            x_train, y_train = load_iter1_data(
                names_lou=names_lou,
                mask_names=mask_names,
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
            flair_name = os.path.join(path, options['flair'])
            image_nii = load_nii(flair_name)
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
            outputname = os.path.join('/'.join(patient[0].rsplit('/')[:-1]), 'test' + str(i) + '.iter1.nii.gz')
            try:
                load_nii(outputname)
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                      c['g'] + '     Patient ' + patient[0].rsplit('/')[-2] + ' already done' + c['nc'])
            except IOError:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                      c['g'] + '     Testing with patient ' + c['b'] + patient[0].rsplit('/')[-2] + c['nc'])
                image_nii = load_nii(patient[0])
                image = np.zeros_like(image_nii.get_data())
                print('    0% of data tested', end='\r')
                for batch, centers, percent in load_patch_batch_percent(patient, 100000, patch_size):
                    y_pred = net.predict_proba(batch)
                    print('    %f%% of data tested' % percent, end='\r')
                    [x, y, z] = np.stack(centers, axis=1)
                    image[x, y, z] = y_pred[:, 1]

                print(c['g'] + '                   -- Saving image ' + c['b'] + outputname + c['nc'])
                image_nii.get_data()[:] = image
                image_nii.to_filename(outputname)

        ''' Here we perform the last iteration '''
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Running iteration ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
        outputname2 = os.path.join(path, 'test' + str(i) + '.new.iter2.nii.gz')
        net_name = os.path.join(path, 'deep-challenge2016.final.new.')
        net = NeuralNet(
            layers=[
                (InputLayer, dict(name='in', shape=(None, 4, 15, 15, 15))),
                (Conv3DDNNLayer, dict(name='conv1_1', num_filters=32, filter_size=(5, 5, 5), pad='same')),
                (Pool3DDNNLayer, dict(name='avgpool_1', pool_size=2, stride=2, mode='average_inc_pad')),
                (Conv3DDNNLayer, dict(name='conv2_1', num_filters=64, filter_size=(5, 5, 5), pad='same')),
                (Pool3DDNNLayer, dict(name='avgpool_2', pool_size=2, stride=2, mode='average_inc_pad')),
                (DropoutLayer, dict(name='l2drop', p=0.5)),
                (DenseLayer, dict(name='l1', num_units=256)),
                (DenseLayer, dict(name='out', num_units=2, nonlinearity=nonlinearities.softmax)),
            ],
            objective_loss_function=objectives.categorical_crossentropy,
            update=updates.adam,
            update_learning_rate=0.0001,
            on_epoch_finished=[
                SaveWeights(net_name + 'model_weights.pkl', only_best=True, pickle=False),
                EarlyStopping(patience=50)
            ],
            batch_iterator_train=BatchIterator(batch_size=4096),
            verbose=10,
            max_epochs=2000,
            train_split=TrainSplit(eval_size=0.25),
            custom_scores=[('dsc', lambda pred, t: 2 * np.sum(pred * t[:, 1]) / np.sum((pred + t[:, 1])))],
        )

        try:
            net.load_params_from(net_name + 'model_weights.pkl')
        except IOError:
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Loading the data for ' + c['b'] + 'iteration 2' + c['nc'])
            names_lou = np.concatenate([names[:, :i], names[:, i + 1:]], axis=1)
            paths = ['/'.join(name.rsplit('/')[:-1]) for name in names_lou[0, :]]
            roi_names = [os.path.join(p_path, 'test' + str(i) + '.iter1.nii.gz') for p_path in paths]
            mask_names = [os.path.join(p_path, mask_name) for p_path in paths]

            x_train, y_train = load_iter2_data(
                names_lou=names_lou,
                mask_names=mask_names,
                roi_names=roi_names,
                patch_size=patch_size,
                seed=seed
            )

            print('              Training vector shape = (' + ','.join([str(length) for length in x_train.shape]) + ')')
            print('              Training labels shape = (' + ','.join([str(length) for length in y_train.shape]) + ')')
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Training (' + c['b'] + 'final' + c['nc'] + c['g'] + ')' + c['nc'])
            net.fit(x_train, y_train)
        try:
            image_nii = load_nii(outputname2)
            image2 = image_nii.get_data()
        except IOError:
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  '<Creating the probability map ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
            image_nii = load_nii(os.path.join(path, flair_b_name))
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
        seg = np.roll(np.roll(image, 1, axis=0), 1, axis=1)
        image_nii.get_data()[:] = seg
        outputname_final = os.path.join(path, 'test' + str(i) + '.old.final.nii.gz') if options['old'] \
            else os.path.join(path, 'test' + str(i) + '.new.final.nii.gz')
        image_nii.to_filename(outputname_final)

        gt = load_nii(os.path.join(path, mask_name)).get_data().astype(dtype=np.bool)
        dsc = np.sum(2.0 * np.logical_and(gt, seg)) / (np.sum(gt) + np.sum(seg))
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<DSC value for ' + c['c'] + case + c['g'] + ' = ' + c['b'] + str(dsc) + c['nc'] + c['g'] + '>' + c['nc'])


if __name__ == '__main__':
    main()
