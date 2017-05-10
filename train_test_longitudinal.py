from __future__ import print_function
import pickle
import argparse
import os
import sys
from time import strftime
import numpy as np
from nets import create_cnn3d_longitudinal, create_cnn3d_det_string, create_cnn_greenspan
from data_creation import load_patch_batch_percent
from data_creation import load_lesion_cnn_data
from nibabel import load as load_nii
from data_manipulation.metrics import dsc_seg, tp_fraction_seg, fp_fraction_seg
from utils import color_codes
from lasagne.layers import DenseLayer


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Subtraction/')
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=9)
    parser.add_argument('-p', '--pool-size', dest='pool_size', type=int, default=2)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=2)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10000)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-u', '--unbalanced', action='store_false', dest='balanced', default=True)
    group.add_argument('-U', '--unbalanced-freeze', action='store_true', dest='freeze', default=False)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-D', '--deformation', dest='deformation', type=int, default=0)
    parser.add_argument('-n', '--num-filters', action='store', dest='number_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-l', '--layers', action='store', dest='layers', default='ca')
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=1000)
    parser.add_argument('--image-folder', dest='image_folder', default='time2/preprocessed/')
    parser.add_argument('--sub-folder', dest='sub_folder', default='time2/subtraction/')
    parser.add_argument('--deformation-folder', dest='defo_folder', default='time2/deformation/')
    parser.add_argument('--no-flair', action='store_false', dest='use_flair', default=True)
    parser.add_argument('--no-pd', action='store_false', dest='use_pd', default=True)
    parser.add_argument('--no-t2', action='store_false', dest='use_t2', default=True)
    parser.add_argument('--flair-baseline', action='store', dest='flair_b', default='flair_moved.nii.gz')
    parser.add_argument('--pd-baseline', action='store', dest='pd_b', default='pd_moved.nii.gz')
    parser.add_argument('--t2-baseline', action='store', dest='t2_b', default='t2_moved.nii.gz')
    parser.add_argument('--flair-sub', action='store', dest='flair_sub', default='flair_smoothed_subtraction.nii.gz')
    parser.add_argument('--pd-sub', action='store', dest='pd_sub', default='pd_smoothed_subtraction.nii.gz')
    parser.add_argument('--t2-sub', action='store', dest='t2_sub', default='t2_smoothed_subtraction.nii.gz')
    parser.add_argument('--flair-12m', action='store', dest='flair_f', default='flair_registered.nii.gz')
    parser.add_argument('--pd-12m', action='store', dest='pd_f', default='pd_corrected.nii.gz')
    parser.add_argument('--t2-12m', action='store', dest='t2_f', default='t2_corrected.nii.gz')
    parser.add_argument('--flair-defo', action='store', dest='flair_d', default='flair_multidemons_deformation.nii.gz')
    parser.add_argument('--pd-defo', action='store', dest='pd_d', default='pd_multidemons_deformation.nii.gz')
    parser.add_argument('--t2-defo', action='store', dest='t2_d', default='t2_multidemons_deformation.nii.gz')
    parser.add_argument('--mask', action='store', dest='mask', default='gt_mask.nii')
    parser.add_argument('--wm-mask', action='store', dest='wm_mask', default='union_wm_mask.nii.gz')
    parser.add_argument('--brain-mask', action='store', dest='brain_mask', default='brainmask.nii.gz')
    parser.add_argument('--padding', action='store', dest='padding', default='valid')
    parser.add_argument('--register', action='store_true', dest='register', default=False)
    parser.add_argument('--greenspan', action='store_true', dest='greenspan', default=False)
    parser.add_argument('-m', '--multi-channel', action='store_true', dest='multi', default=False)
    return vars(parser.parse_args())


def get_names_from_path(path, options, patients=None):
    # Check if all images should be used
    use_flair = options['use_flair']
    use_pd = options['use_pd']
    use_t2 = options['use_t2']

    # Prepare the names for each image
    images_name = options['image_folder']
    flair_b_name = os.path.join(images_name, options['flair_b'])
    pd_b_name = os.path.join(images_name, options['pd_b'])
    t2_b_name = os.path.join(images_name, options['t2_b'])
    flair_f_name = os.path.join(images_name, options['flair_f'])
    pd_f_name = os.path.join(images_name, options['pd_f'])
    t2_f_name = os.path.join(images_name, options['t2_f'])

    # Prepare the names
    if patients:
        flair_b_names = [os.path.join(path, patient, flair_b_name) for patient in patients] if use_flair else None
        pd_b_names = [os.path.join(path, patient, pd_b_name) for patient in patients] if use_pd else None
        t2_b_names = [os.path.join(path, patient, t2_b_name) for patient in patients] if use_t2 else None
        flair_f_names = [os.path.join(path, patient, flair_f_name) for patient in patients] if use_flair else None
        pd_f_names = [os.path.join(path, patient, pd_f_name) for patient in patients] if use_pd else None
        t2_f_names = [os.path.join(path, patient, t2_f_name) for patient in patients] if use_t2 else None
    else:
        flair_b_names = os.path.join(path, flair_b_name) if use_flair else None
        pd_b_names = os.path.join(path, pd_b_name) if use_pd else None
        t2_b_names = os.path.join(path, t2_b_name) if use_t2 else None
        flair_f_names = os.path.join(path, flair_f_name) if use_flair else None
        pd_f_names = os.path.join(path, pd_f_name) if use_pd else None
        t2_f_names = os.path.join(path, t2_f_name) if use_t2 else None
    name_list = [flair_f_names, pd_f_names, t2_f_names, flair_b_names, pd_b_names, t2_b_names]

    return np.stack([name for name in name_list if name is not None])


def get_sub_names_from_path(path, images_folder, sub_name, patients):
    return np.stack([os.path.join(path, patient, images_folder, sub_name) for patient in patients])


def get_defonames_from_path(path, options, patients=None):
    # Check if all images should be used
    use_flair = options['use_flair']
    use_pd = options['use_pd']
    use_t2 = options['use_t2']

    # Prepare the names for each image
    defo_name = options['defo_folder']
    flair_d_name = os.path.join(defo_name, options['flair_d'])
    pd_d_name = os.path.join(defo_name, options['pd_d'])
    t2_d_name = os.path.join(defo_name, options['t2_d'])

    # Prepare the names
    if patients:
        flair_d_names = [os.path.join(path, patient, flair_d_name) for patient in patients] if use_flair else None
        pd_d_names = [os.path.join(path, patient, pd_d_name) for patient in patients] if use_pd else None
        t2_d_names = [os.path.join(path, patient, t2_d_name) for patient in patients] if use_t2 else None
    else:
        flair_d_names = os.path.join(path, flair_d_name) if use_flair else None
        pd_d_names = os.path.join(path, pd_d_name) if use_pd else None
        t2_d_names = os.path.join(path, t2_d_name) if use_t2 else None
    name_list = [flair_d_names, pd_d_names, t2_d_names]

    return np.stack([name for name in name_list if name is not None])


def train_net(
        net,
        x_train,
        y_train,
        images,
        b_name='\033[30mbaseline_%s\033[0m',
        f_name='\033[30mfollow_%s\033[0m',
        d_name='\033[30mdeformation_%s\033[0m'
):
    defo = False
    d_inputs = []
    c = color_codes()
    n_images = len(images)
    # We try to get the last weights to keep improving the net over and over
    if isinstance(x_train, tuple):
        defo = True
        x_train, defo_train = x_train
        defo_train = np.split(defo_train, len(images), axis=1)
        d_inputs = [(d_name % im, np.squeeze(d_im)) for im, d_im in zip(images, defo_train)]
    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Training' + c['nc'])
    n_channels = x_train.shape[1]
    x_train = np.split(x_train, n_channels, axis=1)
    b_inputs = [(b_name % im, x_im) for im, x_im in zip(images, x_train[:n_images])]
    f_inputs = [(f_name % im, x_im) for im, x_im in zip(images, x_train[n_images:])]
    inputs = dict(b_inputs + f_inputs) if not defo else dict(b_inputs + f_inputs + d_inputs)
    net.fit(inputs, y_train)


def train_greenspan(
        net,
        x_train,
        y_train,
        images,
        b_name='\033[30mbaseline_%s\033[0m',
        f_name='\033[30mfollow_%s\033[0m'
):
        c = color_codes()
        n_axis = x_train.shape[1]
        n_images = x_train.shape[2]/2
        print('                Training vector shape ='
              ' (' + ','.join([str(length) for length in x_train.shape]) + ')')
        print('                Training labels shape ='
              ' (' + ','.join([str(length) for length in y_train.shape]) + ')')
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Training' + c['nc'])
        # We try to get the last weights to keep improving the net over and over
        x_train = np.split(x_train, n_axis, axis=1)
        b_inputs = [(b_name % im, np.squeeze(x_im[:, :, :n_images, :, :])) for im, x_im in zip(images, x_train)]
        f_inputs = [(f_name % im, np.squeeze(x_im[:, :, n_images:, :, :])) for im, x_im in zip(images, x_train)]
        inputs = dict(b_inputs + f_inputs)
        net.fit(inputs, y_train)


def test_net(
        net,
        names,
        mask,
        batch_size,
        patch_size,
        defo_size,
        image_size,
        images,
        d_names=None,
        b_name='\033[30mbaseline_%s\033[0m',
        f_name='\033[30mfollow_%s\033[0m',
        d_name='\033[30mdeformation_%s\033[0m'
):
    defo = False
    d_inputs = []
    n_images = len(images)
    n_channels = n_images * 2
    test = np.zeros(image_size)
    print('              0% of data tested', end='\r')
    sys.stdout.flush()
    for batch, centers, percent in load_patch_batch_percent(
            names,
            batch_size,
            patch_size,
            defo_size,
            d_names=d_names,
            mask=mask
    ):
        if isinstance(batch, tuple):
            defo = True
            batch, d_batch = batch
            d_batch = np.split(d_batch, len(images), axis=1)
            d_inputs = [(d_name % im, np.squeeze(d_im)) for im, d_im in zip(images, d_batch)]
        batch = np.split(batch, n_channels, axis=1)
        b_inputs = [(b_name % im, x_im) for im, x_im in zip(images, batch[:n_images])]
        f_inputs = [(f_name % im, x_im) for im, x_im in zip(images, batch[n_images:])]
        inputs = dict(b_inputs + f_inputs) if not defo else dict(b_inputs + f_inputs + d_inputs)
        y_pred = net.predict_proba(inputs)
        print('              %f%% of data tested' % percent, end='\r')
        sys.stdout.flush()
        [x, y, z] = np.stack(centers, axis=1)
        test[x, y, z] = y_pred[:, -1]

    return test


def test_greenspan(
            net,
            names,
            mask,
            batch_size,
            patch_size,
            image_size,
            images,
            b_name='\033[30mbaseline_%s\033[0m',
            f_name='\033[30mfollow_%s\033[0m'
):

    n_axis = len(images)
    n_images = len(names) / 2
    test = np.zeros(image_size)
    print('              0% of data tested', end='\r')
    sys.stdout.flush()
    for batch, centers, percent in load_patch_batch_percent(names, batch_size, patch_size, mask=mask):
        batch = np.split(np.swapaxes(batch, 0, 2), n_axis, axis=1)
        b_inputs = [(b_name % im, np.squeeze(x_im[:, :, :n_images, :, :])) for im, x_im in zip(images, batch)]
        f_inputs = [(f_name % im, np.squeeze(x_im[:, :, n_images:, :, :])) for im, x_im in zip(images, batch)]
        inputs = dict(b_inputs + f_inputs)
        y_pred = net.predict_proba(inputs)
        print('              %f%% of data tested' % percent, end='\r')
        sys.stdout.flush()
        [x, y, z] = np.stack(centers, axis=1)
        test[x, y, z] = y_pred[:, 1]

    return test


def main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the net architecture parameters
    register = options['register']
    multi = options['multi']
    defo = options['deformation']
    layers = ''.join(options['layers'])
    greenspan = options['greenspan']
    freeze = options['freeze']
    balanced = options['balanced'] if not freeze else False

    # Prepare the net hyperparameters
    epochs = options['epochs']
    padding = options['padding']
    patch_width = options['patch_width']
    patch_size = (32, 32) if greenspan else (patch_width, patch_width, patch_width)
    pool_size = options['pool_size']
    batch_size = options['batch_size']
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['number_filters']
    n_filters = n_filters if len(n_filters) > 1 else n_filters*conv_blocks
    conv_width = options['conv_width']
    conv_size = conv_width if isinstance(conv_width, list) else [conv_width]*conv_blocks

    # Prepare the sufix that will be added to the results for the net and images
    use_flair = options['use_flair']
    use_pd = options['use_pd']
    use_t2 = options['use_t2']
    flair_name = 'flair' if use_flair else None
    pd_name = 'pd' if use_pd else None
    t2_name = 't2' if use_t2 else None
    images = [name for name in [flair_name, pd_name, t2_name] if name is not None]
    reg_s = '.reg' if register else ''
    filters_s = 'n'.join(['%d' % nf for nf in n_filters])
    conv_s = 'c'.join(['%d' % cs for cs in conv_size])
    im_s = '.'.join(images)
    mc_s = '.mc' if multi else ''
    d_s = 'd%d.' % (conv_blocks*2+defo) if defo else ''
    sufix = '.greenspan' if greenspan else '%s.%s%s%s.p%d.c%s.n%s.d%d.e%d.pad_%s' %\
        (mc_s, d_s, im_s, reg_s, patch_width, conv_s, filters_s, dense_size, epochs, padding)

    # Prepare the data names
    mask_name = options['mask']
    wm_name = options['wm_mask']
    sub_folder = options['sub_folder']
    sub_name = options['flair_sub']
    dir_name = options['dir_name']
    patients = [f for f in sorted(os.listdir(dir_name))
                if os.path.isdir(os.path.join(dir_name, f))]
    n_patients = len(patients)
    names = get_names_from_path(dir_name, options, patients)
    defo_names = get_defonames_from_path(dir_name, options, patients) if defo else None
    defo_width = conv_blocks*2+defo if defo else None
    defo_size = (defo_width, defo_width, defo_width)

    # Random initialisation
    seed = np.random.randint(np.iinfo(np.int32).max)

    # Metrics output
    metrics_file = os.path.join(dir_name, 'metrics' + sufix)

    with open(metrics_file, 'w') as f:

        print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting leave-one-out' + c['nc'])
        # Leave-one-out main loop (we'll do 2 training iterations with testing for each patient)
        for i in range(0, n_patients):
            # Prepare the data relevant to the leave-one-out (subtract the patient from the dataset and set the path)
            # Also, prepare the network
            case = patients[i]
            path = os.path.join(dir_name, case)
            names_lou = np.concatenate([names[:, :i], names[:, i + 1:]], axis=1)
            defo_names_lou = np.concatenate([defo_names[:, :i], defo_names[:, i + 1:]], axis=1) if defo else None
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Patient ' + c['b'] + case + c['nc'] +
                  c['g'] + ' (%d/%d)' % (i+1, n_patients))
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  '<Running iteration ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
            net_name = os.path.join(path, 'deep-longitudinal.init' + sufix + '.')
            if greenspan:
                net = create_cnn_greenspan(
                    input_channels=names.shape[0]/2,
                    patience=25,
                    name=net_name,
                    epochs=500
                )
                images = ['axial', 'coronal', 'sagital']
            else:
                if multi:
                    net = create_cnn3d_det_string(
                        cnn_path=layers,
                        input_shape=(None, names.shape[0], patch_width, patch_width, patch_width),
                        convo_size=conv_size,
                        padding=padding,
                        dense_size=dense_size,
                        pool_size=2,
                        number_filters=n_filters,
                        patience=10,
                        multichannel=True,
                        name=net_name,
                        epochs=100
                    )
                else:
                    net = create_cnn3d_longitudinal(
                        convo_blocks=conv_blocks,
                        input_shape=(None, names.shape[0], patch_width, patch_width, patch_width),
                        images=images,
                        convo_size=conv_size,
                        pool_size=pool_size,
                        dense_size=dense_size,
                        number_filters=n_filters,
                        padding=padding,
                        drop=0.5,
                        register=register,
                        defo=defo,
                        patience=10,
                        name=net_name,
                        epochs=100
                    )

            names_test = get_names_from_path(path, options)
            defo_names_test = get_defonames_from_path(path, options) if defo else None
            outputname1 = os.path.join(path, 't' + case + sufix + '.iter1.nii.gz') if not greenspan else os.path.join(
                path, 't' + case + sufix + '.nii.gz')

            # First we check that we did not train for that patient, in order to save time
            try:
                net.load_params_from(net_name + 'model_weights.pkl')
            except IOError:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                      c['g'] + 'Loading the data for ' + c['b'] + 'iteration 1' + c['nc'])
                # Data loading. Most of it is based on functions from data_creation that load the data.
                #  But we also need to prepare the name list to load the leave-one-out data.
                paths = [os.path.join(dir_name, p) for p in np.concatenate([patients[:i], patients[i+1:]])]
                mask_names = [os.path.join(p_path, mask_name) for p_path in paths]
                wm_names = [os.path.join(p_path, wm_name) for p_path in paths]
                pr_names = [os.path.join(p_path, sub_folder, sub_name) for p_path in paths]

                x_train, y_train = load_lesion_cnn_data(
                    names=names_lou,
                    mask_names=mask_names,
                    defo_names=defo_names_lou,
                    roi_names=wm_names,
                    pr_names=pr_names,
                    patch_size=patch_size,
                    defo_size=defo_size,
                    random_state=seed
                )

                # Afterwards we train. Check the relevant training function.
                if greenspan:
                    x_train = np.swapaxes(x_train, 1, 2)
                    train_greenspan(net, x_train, y_train, images)
                else:
                    train_net(net, x_train, y_train, images)
                    with open(net_name + 'layers.pkl', 'wb') as fnet:
                        pickle.dump(net.layers, fnet, -1)
            # Then we test the net. Again we save time by checking if we already tested that patient.
            try:
                image_nii = load_nii(outputname1)
                image1 = image_nii.get_data()
            except IOError:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                      '<Creating the probability map ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
                image_nii = load_nii(os.path.join(path, options['image_folder'], options['flair_f']))
                mask_nii = load_nii(os.path.join(path, wm_name))
                if greenspan:
                    image1 = test_greenspan(
                        net,
                        names_test,
                        mask_nii.get_data(),
                        batch_size,
                        patch_size,
                        image_nii.get_data().shape,
                        images
                    )
                else:
                    image1 = test_net(
                        net,
                        names_test,
                        mask_nii.get_data(),
                        batch_size,
                        patch_size,
                        defo_size,
                        image_nii.get_data().shape,
                        images,
                        defo_names_test
                    )
                image_nii.get_data()[:] = image1
                image_nii.to_filename(outputname1)
            if greenspan:
                # Since Greenspan did not use two iterations, we must get the final mask here.
                outputname_final = os.path.join(path, 't' + case + sufix + '.final.nii.gz')
                mask_nii.get_data()[:] = (image1 > 0.5).astype(dtype=np.int8)
                mask_nii.to_filename(outputname_final)
            else:
                # If not, we test the net with the training set to look for misclassified negative with a high
                # probability of being positives according to the net.
                # These voxels will be the input of the second training iteration.
                ''' Here we get the seeds '''
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                      c['g'] + '<Looking for seeds for the final iteration>' + c['nc'])
                patients_names = zip(np.rollaxis(names_lou, 1), np.rollaxis(defo_names_lou, 1)) if defo\
                    else np.rollaxis(names_lou, 1)
                for patient in patients_names:
                    if defo:
                        patient, d_patient = patient
                    else:
                        d_patient = None
                    patient_path = '/'.join(patient[0].rsplit('/')[:-1])
                    outputname = os.path.join(patient_path, 't' + case + sufix + '.nii.gz')
                    mask_nii = load_nii(os.path.join('/'.join(patient[0].rsplit('/')[:-3]), wm_name))
                    try:
                        load_nii(outputname)
                        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                              c['g'] + '     Patient ' + patient[0].rsplit('/')[-4] + ' already done' + c['nc'])
                    except IOError:
                        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                              c['g'] + '     Testing with patient ' + c['b'] + patient[0].rsplit('/')[-4] + c['nc'])
                        image_nii = load_nii(patient[0])

                        image = test_net(
                            net,
                            patient,
                            mask_nii.get_data(),
                            batch_size,
                            patch_size,
                            defo_size,
                            image_nii.get_data().shape,
                            images,
                            d_patient
                        )

                        print(c['g'] + '                   -- Saving image ' + c['b'] + outputname + c['nc'])
                        image_nii.get_data()[:] = image
                        image_nii.to_filename(outputname)

                ''' Here we perform the last iteration '''
                # Finally we perform the final iteration. After refactoring the code, the code looks almost exactly
                # the same as the training of the first iteration.
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                      '<Running iteration ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
                f_s = '.f' if freeze else ''
                ub_s = '.ub' if not balanced else ''
                final_s = f_s + ub_s
                outputname2 = os.path.join(path, 't' + case + final_s + sufix + '.iter2.nii.gz')
                net_name = os.path.join(path, 'deep-longitudinal.final' + final_s + sufix + '.')
                if multi:
                    net = create_cnn3d_det_string(
                        cnn_path=layers,
                        input_shape=(None, names.shape[0], patch_width, patch_width, patch_width),
                        convo_size=conv_size,
                        padding=padding,
                        pool_size=2,
                        dense_size=dense_size,
                        number_filters=n_filters,
                        patience=50,
                        multichannel=True,
                        name=net_name,
                        epochs=epochs
                    )
                else:
                    if not freeze:
                        net = create_cnn3d_longitudinal(
                            convo_blocks=conv_blocks,
                            input_shape=(None, names.shape[0], patch_width, patch_width, patch_width),
                            images=images,
                            convo_size=conv_size,
                            pool_size=pool_size,
                            dense_size=dense_size,
                            number_filters=n_filters,
                            padding=padding,
                            drop=0.5,
                            register=register,
                            defo=defo,
                            patience=50,
                            name=net_name,
                            epochs=epochs
                        )
                    else:
                        net.max_epochs=epochs
                        for layer in net.get_all_layers():
                            if not isinstance(layer, DenseLayer):
                                for param in layer.params:
                                    layer.params[param].discard('trainable')

                try:
                    net.load_params_from(net_name + 'model_weights.pkl')
                except IOError:
                    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                          c['g'] + 'Loading the data for ' + c['b'] + 'iteration 2' + c['nc'])
                    roi_paths = ['/'.join(name.rsplit('/')[:-1]) for name in names_lou[0, :]]
                    paths = [os.path.join(dir_name, p) for p in np.concatenate([patients[:i], patients[i + 1:]])]
                    ipr_names = [os.path.join(p_path, sub_folder, sub_name) for p_path in paths] if freeze else None
                    pr_names = [os.path.join(p_path, 't' + case + sufix + '.nii.gz') for p_path in roi_paths]
                    mask_names = [os.path.join(p_path, mask_name) for p_path in paths]
                    wm_names = [os.path.join(p_path, wm_name) for p_path in paths]

                    x_train, y_train = load_lesion_cnn_data(
                        names=names_lou,
                        mask_names=mask_names,
                        defo_names=defo_names_lou,
                        roi_names=wm_names,
                        init_pr_names=ipr_names,
                        pr_names=pr_names,
                        patch_size=patch_size,
                        defo_size=defo_size,
                        random_state=seed,
                        balanced=balanced
                    )

                    train_net(net, x_train, y_train, images)
                    with open(net_name + 'layers.pkl', 'wb') as fnet:
                        pickle.dump(net.layers, fnet, -1)
                try:
                    image_nii = load_nii(outputname2)
                    image2 = image_nii.get_data()
                except IOError:
                    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                          '<Creating the probability map ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
                    image_nii = load_nii(os.path.join(path, options['image_folder'], options['flair_f']))
                    mask_nii = load_nii(os.path.join(path, wm_name))
                    image2 = test_net(
                        net,
                        names_test,
                        mask_nii.get_data(),
                        batch_size,
                        patch_size,
                        defo_size,
                        image_nii.get_data().shape,
                        images,
                        defo_names_test
                    )

                    image_nii.get_data()[:] = image2
                    image_nii.to_filename(outputname2)

                image = image1 * image2
                image_nii.get_data()[:] = image
                outputname_mult = os.path.join(path, 't' + case + final_s + sufix + '.iter1_x_2.nii.gz')
                image_nii.to_filename(outputname_mult)

                image = (image1 * image2) > 0.5
                image_nii.get_data()[:] = image
                outputname_final = os.path.join(path, 't' + case + final_s + sufix + '.final.nii.gz')
                image_nii.to_filename(outputname_final)

            # Finally we compute some metrics that are stored in the metrics file defined above.
            # I plan on replicating Challenge's 2008 evaluation measures here.
            gt = load_nii(os.path.join(path, mask_name)).get_data().astype(dtype=np.bool)
            seg1 = image1 > 0.5
            if not greenspan:
                seg2 = image2 > 0.5
            dsc1 = dsc_seg(gt, seg1)
            if not greenspan:
                dsc2 = dsc_seg(gt, seg2)
            if not greenspan:
                dsc_final = dsc_seg(gt, image)
            else:
                dsc_final = dsc1
            tpf1 = tp_fraction_seg(gt, seg1)
            if not greenspan:
                tpf2 = tp_fraction_seg(gt, seg2)
            if not greenspan:
                tpf_final = tp_fraction_seg(gt, image)
            fpf1 = fp_fraction_seg(gt, seg1)
            if not greenspan:
                fpf2 = fp_fraction_seg(gt, seg2)
            if not greenspan:
                fpf_final = fp_fraction_seg(gt, image)
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  '<DSC ' + c['c'] + case + c['g'] + ' = ' + c['b'] + str(dsc_final) + c['nc'] + c['g'] + '>' + c['nc'])
            f.write('%s;Test 1; %f;%f;%f\n' % (case, dsc1, tpf1, fpf1))
            if not greenspan:
                f.write('%s;Test 2; %f;%f;%f\n' % (case, dsc2, tpf2, fpf2))
            if not greenspan:
                f.write('%s;Final; %f;%f;%f\n' % (case, dsc_final, tpf_final, fpf_final))


if __name__ == '__main__':
    main()
