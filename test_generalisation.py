from __future__ import print_function
import pickle
import argparse
import os
from time import strftime
import numpy as np
from nets import create_cnn3d_longitudinal
from data_creation import load_lesion_cnn_data
from data_creation import save_nifti
from nibabel import load as load_nii
# from data_manipulation.metrics import dsc_seg, tp_fraction_seg, fp_fraction_seg
from scipy.ndimage.interpolation import zoom
from utils import color_codes, WeightsLogger
from train_test_longitudinal import get_defonames_from_path, get_names_from_path, test_net, train_net
import itertools


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test generalisation of nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Generalisation/')
    parser.add_argument('-p', '--pool-size', dest='pool_size', type=int, default=1)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', type=int, default=3)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10000)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=500)
    parser.add_argument('-l', '--last-width', action='store', dest='last_width', type=int, default=3)
    parser.add_argument('--image-folder', dest='image_folder', default='time2/preprocessed/')
    parser.add_argument('--deformation-folder', dest='defo_folder', default='time2/deformation/')
    parser.add_argument('--flair-baseline', action='store', dest='flair_b', default='flair_moved.nii.gz')
    parser.add_argument('--pd-baseline', action='store', dest='pd_b', default='pd_moved.nii.gz')
    parser.add_argument('--t2-baseline', action='store', dest='t2_b', default='t2_moved.nii.gz')
    parser.add_argument('--flair-12m', action='store', dest='flair_f', default='flair_registered.nii.gz')
    parser.add_argument('--pd-12m', action='store', dest='pd_f', default='pd_corrected.nii.gz')
    parser.add_argument('--t2-12m', action='store', dest='t2_f', default='t2_corrected.nii.gz')
    parser.add_argument('--flair-defo', action='store', dest='flair_d', default='flair_multidemons_deformation.nii.gz')
    parser.add_argument('--pd-defo', action='store', dest='pd_d', default='pd_multidemons_deformation.nii.gz')
    parser.add_argument('--t2-defo', action='store', dest='t2_d', default='t2_multidemons_deformation.nii.gz')
    parser.add_argument('--mask', action='store', dest='mask', default='gt_mask.nii')
    parser.add_argument('--wm-mask', action='store', dest='wm_mask', default='union_wm_mask.nii.gz')
    parser.add_argument('--brain-mask', action='store', dest='brain_mask', default='brainmask.nii.gz')
    return vars(parser.parse_args())


def train_all_nets(
        names_lou,
        defo_names_lou,
        mask_names,
        roi_names,
        net_names,
        conv_blocks,
        patch_sizes,
        defo_sizes,
        conv_sizes,
        n_filters,
        images,
        pool_size,
        dense_sizes,
        epochs,
        seed,
):
    # We need to prepare the name list to load the leave-one-out data.
    # Since these names are common for all the nets, we can define them before looping.
    c = color_codes()

    net_combos = itertools.product(zip(conv_blocks, patch_sizes, conv_sizes, defo_sizes), n_filters, dense_sizes)

    nets = list()

    # Data loading. We load the largest possible patch size, and then we resize everything
    max_patch = max(patch_sizes)
    max_defo = max(defo_sizes)
    x_train, y_train = load_lesion_cnn_data(
        names=names_lou,
        mask_names=mask_names,
        defo_names=defo_names_lou,
        roi_names=roi_names,
        pr_names=None,
        patch_size=max_patch,
        defo_size=max_defo,
        random_state=seed
    )

    for ((blocks, patch, convo, defo), filters, dense), net_name in zip(net_combos, net_names):

        net = create_cnn3d_longitudinal(
            convo_blocks=blocks,
            input_shape=(None, 6) + patch,
            images=images,
            convo_size=convo,
            pool_size=pool_size,
            dense_size=dense,
            number_filters=filters,
            padding='valid',
            drop=0.5,
            register=False,
            defo=True,
            patience=epochs,
            name=net_name,
            epochs=epochs
        )
        # First we check that we did not train that patient, in order to save time
        try:
            net.load_params_from(net_name + 'model_weights.pkl')
        except IOError:
            combo_s = ' [c%d.k%d.d%d.n%d]' % (
                blocks,
                convo[0],
                dense,
                filters
            )

            # Afterwards we train. Check the relevant training function.
            # NOTE: We have to resize the training patches if necessary.
            patch_ratio = (1, 1) + tuple(itertools.imap(lambda x, y: float(x) / y, patch, max_patch))
            defo_ratio = (1, 1, 1) + tuple(itertools.imap(lambda x, y: float(x) / y, defo, max_defo))

            print(c['c'] + '[' + strftime("%H:%M:%S") + ']      ' + c['g'] + 'Patch shape = (' +
                  ','.join([c['bg'] + str(length) + c['nc'] + c['g'] for length in patch]) + ')' +
                  combo_s + c['nc'])
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']      ' + c['g'] + 'Deformation shape = (' +
                  ','.join([c['bg'] + str(length) + c['nc'] + c['g'] for length in defo]) + ')' +
                  combo_s + c['nc'])

            train_net(
                net=net,
                x_train=(zoom(x_train[0], patch_ratio), zoom(x_train[1], defo_ratio)),
                y_train=y_train,
                images=images
            ) if max_patch != patch else train_net(
                net=net,
                x_train=x_train,
                y_train=y_train,
                images=images
            )
            for callback in net.on_epoch_finished:
                if isinstance(callback, WeightsLogger):
                    callback.save()
            with open(net_name + 'layers.pkl', 'wb') as fnet:
                pickle.dump(net.layers, fnet, -1)
        nets.append(net)

    return nets


def test_all_nets(
        path,
        names_test,
        defo_names_test,
        roi_name,
        nets,
        case,
        batch_size,
        patch_sizes,
        defo_sizes,
        dense_sizes,
        n_filters,
        sufixes,
        iter_name,
        train_case=False
):
    c = color_codes()
    net_combos = itertools.product(zip(patch_sizes, defo_sizes), n_filters, dense_sizes)

    image_nii = load_nii(names_test[0])
    mask_nii = load_nii(roi_name)

    images = list()

    for net, sufix, ((patch_size, defo_size), _, _) in zip(nets, sufixes, net_combos):
        outputname = os.path.join(path, 't' + case + sufix + iter_name + '.nii.gz')
        # We save time by checking if we already tested that patient.
        try:
            image_nii = load_nii(outputname)
            image = image_nii.get_data()
            if train_case:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']      ' +
                      c['g'] + '     Patient ' + names_test[0].rsplit('/')[-4] + ' already done' + c['nc'])
        except IOError:
            if train_case:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']      ' +
                      c['g'] + '     Testing with patient ' + c['b'] + names_test[0].rsplit('/')[-4] + c['nc'])
            else:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']      ' + c['g'] +
                      '<Creating the probability map for net: ' +
                      c['b'] + sufix + c['nc'] + c['g'] + '>' + c['nc'])
            image = test_net(
                net=net,
                names=names_test,
                mask=mask_nii.get_data(),
                batch_size=batch_size,
                patch_size=patch_size,
                defo_size=defo_size,
                image_size=image_nii.get_data().shape,
                images=['flair', 'pd', 't2'],
                d_names=defo_names_test
            )

            if train_case:
                print(c['g'] + '                     -- Saving image ' + c['b'] + outputname + c['nc'])
            image_nii.get_data()[:] = image
            image_nii.to_filename(outputname)

        images.append(image)

    return images


def main():
    options = parse_inputs()
    options['use_flair'] = True
    options['use_pd'] = True
    options['use_t2'] = True
    conv_blocks = [1, 2, 3]

    c = color_codes()

    # Prepare the net hyperparameters
    epochs = options['epochs']
    conv_width = options['conv_width']
    conv_sizes = [[conv_width]*blocks for blocks in conv_blocks]
    last_width = options['last_width']
    patch_widths = [blocks * (conv_width - 1) + last_width for blocks in conv_blocks]
    patch_sizes = [(width, width, width) for width in patch_widths]
    pool_size = 1
    batch_size = 100000
    dense_sizes = [16, 64, 128, 256]
    n_filters = [32, 64]

    # Prepare the sufix that will be added to the results for the net and images
    flair_name = 'flair'
    pd_name = 'pd'
    t2_name = 't2'
    images = [name for name in [flair_name, pd_name, t2_name] if name is not None]
    parameter_combos = itertools.product(patch_widths, conv_sizes, n_filters, dense_sizes)
    name_parameters = [(
                           combo[0],
                           'c'.join([str(cs) for cs in combo[1]]),
                           combo[2],
                           combo[3],
                           epochs
                       ) for combo in parameter_combos]
    sufixes = ['.p%d.c%s.n%s.d%d.e%d' % p for p in name_parameters]

    # Prepare the data names
    mask_name = options['mask']
    wm_name = options['wm_mask']
    dir_name = options['dir_name']
    patients = [f for f in sorted(os.listdir(dir_name))
                if os.path.isdir(os.path.join(dir_name, f))]
    n_patients = len(patients)
    names = get_names_from_path(dir_name, options, patients)
    defo_names = get_defonames_from_path(dir_name, options, patients)
    defo_widths = [blocks * 2 + 1 for blocks in conv_blocks]
    defo_sizes = [(width, width, width) for width in defo_widths]

    # Random initialisation
    seed = np.random.randint(np.iinfo(np.int32).max)

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting leave-one-out' + c['nc'])
    # Leave-one-out main loop (we'll do 2 training iterations with testing for each patient)
    for i in range(0, n_patients):
        # Prepare the data relevant to the leave-one-out (subtract the patient from the dataset and set the path)
        # Also, prepare the network

        case = patients[i]
        path = os.path.join(dir_name, case)
        names_lou = np.concatenate([names[:, :i], names[:, i + 1:]], axis=1)
        defo_names_lou = np.concatenate([defo_names[:, :i], defo_names[:, i + 1:]], axis=1)

        paths = [os.path.join(dir_name, p) for p in np.concatenate([patients[:i], patients[i + 1:]])]
        mask_names = [os.path.join(p_path, mask_name) for p_path in paths]
        wm_names = [os.path.join(p_path, wm_name) for p_path in paths]

        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Patient ' + c['b'] + case + c['nc'] +
              c['g'] + ' (%d/%d)' % (i+1, n_patients))
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Running iteration ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])

        net_names = [os.path.join(path, 'deep-generalise.init' + sufix + '.') for sufix in sufixes]
        nets = train_all_nets(
            names_lou=names_lou,
            defo_names_lou=defo_names_lou,
            mask_names=mask_names,
            roi_names=wm_names,
            net_names=net_names,
            conv_blocks=conv_blocks,
            patch_sizes=patch_sizes,
            defo_sizes=defo_sizes,
            conv_sizes=conv_sizes,
            n_filters=n_filters,
            images=images,
            pool_size=pool_size,
            dense_sizes=dense_sizes,
            epochs=epochs,
            seed=seed
        )

        # Then we test the net.
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']      ' + c['g'] + '<Testing iteration ' + c['b'] + '1' +
              c['nc'] + c['g'] + '>' + c['nc'])
        names_test = get_names_from_path(path, options)
        defo_names_test = get_defonames_from_path(path, options)
        roi_name = os.path.join(path, wm_name)
        test_all_nets(
            path=path,
            names_test=names_test,
            defo_names_test=defo_names_test,
            roi_name=roi_name,
            nets=nets,
            case=case,
            batch_size=batch_size,
            patch_sizes=patch_sizes,
            defo_sizes=defo_sizes,
            dense_sizes=dense_sizes,
            n_filters=n_filters,
            sufixes=sufixes,
            iter_name='.generalise'
        )


if __name__ == '__main__':
    main()