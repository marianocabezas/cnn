import os
import argparse
import numpy as np
from data_creation import load_patches, load_patch_batch
from data_creation import get_sufix
from data_creation import leave_one_out
from data_creation import sum_patches_to_image
from nets import create_unet3d_det_string, create_unet3d_shortcuts_det_string
from nets import create_unet3d_seg_string, create_unet3d_shortcuts_seg_string
from nets import create_cnn3d_det_string
from nibabel import load as load_nii


def main():
    # Parse command line options
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='folder', default='/home/mariano/DATA/Challenge/')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    parser.add_argument('-c', '--convolution-size', action='store', dest='convo_size', type=int, default=3)
    parser.add_argument('-p', '--pool-size', action='store', dest='pool_size', type=int, default=2)
    parser.add_argument('-t', '--test-size', action='store', dest='test_size', type=float, default=0.25)
    parser.add_argument('-n', '--number-filters', action='store', dest='number_filters', type=int, default=4)
    parser.add_argument('-l', '--forward-layers', action='store', dest='layers', default='cac')
    parser.add_argument('-w', '--win-size', action='store', dest='patch_size', type=int, nargs=3, default=(9, 9, 9))
    parser.add_argument('-i', '--image-size', action='store', dest='min_shape', type=int, nargs=3, default=None)
    parser.add_argument('-b', '--batch-size', action='store', dest='batch_size', type=int, default=200000)
    parser.add_argument('--patience', action='store', dest='patience', default=20)
    parser.add_argument('--multi-channel', action='store_true', dest='multi_channel', default=True)
    parser.add_argument('--single-channel', action='store_false', dest='multi_channel', default=True)
    parser.add_argument('--use-gado', action='store_true', dest='use_gado', default=False)
    parser.add_argument('--no-gado', action='store_false', dest='use_gado', default=False)
    parser.add_argument('--gado', action='store', dest='gado', default='GADO_preprocessed.nii.gz')
    parser.add_argument('--use-flair', action='store_true', dest='use_flair', default=True)
    parser.add_argument('--no-flair', action='store_false', dest='use_flair', default=True)
    parser.add_argument('--flair', action='store', dest='flair', default='FLAIR_preprocessed.nii.gz')
    parser.add_argument('--use-pd', action='store_true', dest='use_pd', default=True)
    parser.add_argument('--no-pd', action='store_false', dest='use_pd', default=True)
    parser.add_argument('--pd', action='store', dest='pd', default='DP_preprocessed.nii.gz')
    parser.add_argument('--use-t2', action='store_true', dest='use_t2', default=True)
    parser.add_argument('--no-t2', action='store_false', dest='use_t2', default=True)
    parser.add_argument('--t2', action='store', dest='t2', default='T2_preprocessed.nii.gz')
    parser.add_argument('--use-t1', action='store_true', dest='use_t1', default=True)
    parser.add_argument('--no-t1', action='store_false', dest='use_t1', default=True)
    parser.add_argument('--t1', action='store', dest='t1', default='T1_preprocessed.nii.gz')
    parser.add_argument('--mask', action='store', dest='mask', default='Consensus.nii.gz')
    parser.add_argument('--patches-det', action='store_const', const='patches-det', dest='select', default='unet')
    parser.add_argument('--patches-short', action='store_const', const='patches-short', dest='select', default='unet')
    parser.add_argument('--patches-cnn', action='store_const', const='patches-cnn', dest='select', default='unet')

    args = parser.parse_args()

    selector = {
        'patches-det': unet_patches3d_detection,
        'patches-seg': unet_patches3d_segmentation,
        'patches-short': unet_patches3d_shortcuts_detection,
        'patches-short-seg': unet_patches3d_shortcuts_segmentation,
        'patches-cnn': cnn_patches3d_detection,
    }

    options = vars(args)

    selector[options['select']](options)


def color_codes():
    codes = {'g': '\033[32m',
             'c': '\033[36m',
             'bg': '\033[32;1m',
             'b': '\033[1m',
             'nc': '\033[0m',
             'gc': '\033[32m, \033[0m'
             }
    return codes


def unet_patches3d_detection(options):
    patches_network_detection(options, 'unet')


def unet_patches3d_segmentation(options):
    patches_network_segmentation(options, 'unet')


def unet_patches3d_shortcuts_detection(options):
    patches_network_detection(options, 'unet-short')


def unet_patches3d_shortcuts_segmentation(options):
    patches_network_segmentation(options, 'unet-short')


def cnn_patches3d_detection(options):
    patches_network_detection(options, 'cnn')


def patches_network_detection(options, mode):
    c = color_codes()
    image_sufix = get_sufix(
        options['use_flair'],
        options['use_pd'],
        options['use_t2'],
        options['use_gado'],
        options['use_t1']
    )
    size_sufix = '.'.join([str(length) for length in tuple(options['patch_size'])])
    sufixes = image_sufix + '.' + size_sufix
    mode_write = mode + '.mc' if options['multi_channel'] else mode + '.sc'

    print(c['g'] + 'Loading the data for the patch-based ' + c['b'] + mode + c['nc'])
    # Create the data
    (x, y, names) = load_patches(
        dir_name=options['folder'],
        use_flair=options['use_flair'],
        use_pd=options['use_pd'],
        use_t2=options['use_t2'],
        use_gado=options['use_gado'],
        use_t1=options['use_t1'],
        flair_name=options['flair'],
        pd_name=options['pd'],
        t2_name=options['t2'],
        gado_name=options['gado'],
        t1_name=options['t1'],
        mask_name=options['mask'],
        size=tuple(options['patch_size'])
    )

    print(c['g'] + 'Starting leave-one-out for the patch-based ' + c['b'] + mode + c['nc'])

    n_channels = x[0].shape[1]
    channels = range(0, n_channels)

    for x_train, y_train, i in leave_one_out(x, y):
        print('Running patient ' + c['c'] + names[0, i].rsplit('/')[-2] + c['nc'])
        seed = np.random.randint(np.iinfo(np.int32).max)
        print('-- Permuting the data')
        np.random.seed(seed)
        x_train = np.random.permutation(np.concatenate(x_train).astype(dtype=np.float32))
        print('-- Permuting the labels')
        np.random.seed(seed)
        y_train = np.random.permutation(np.concatenate(y_train).astype(dtype=np.int32))
        y_train = y_train[:, y_train.shape[1] / 2, y_train.shape[2] / 2, y_train.shape[3] / 2]
        print('-- Training vector shape = (' + ','.join([str(length) for length in x_train.shape]) + ')')
        print('-- Training labels shape = (' + ','.join([str(length) for length in y_train.shape]) + ')')

        print(c['g'] + '-- Creating the ' + c['b'] + 'patch-based ' + c['b'] + mode + c['nc'])

        # Train the net and save it
        net_name = os.path.join(
            os.path.split(names[0, i])[0], 'patches_' + mode + '.c' + str(i) + '.' + sufixes
        )
        net_types = {
            'cnn': create_cnn3d_det_string,
            'unet': create_unet3d_det_string,
            'unet-short': create_unet3d_shortcuts_det_string
        }
        net = net_types[mode](
            ''.join(options['layers']),
            x_train.shape,
            options['convo_size'],
            options['pool_size'],
            options['number_filters'],
            options['patience'],
            options['multi_channel'],
            net_name
        )

        print(c['g'] + '-- Training the ' + c['b'] + 'patch-based ' + c['b'] + mode + c['nc'])
        # We try to get the last weights to keep improving the net over and over
        try:
            net.load_params_from(net_name + 'model_weights.pkl')
        except IOError:
            pass

        if options['multi_channel']:
            net.fit(x_train, y_train)
        else:
            x_train = np.split(x_train, n_channels, axis=1)
            inputs = dict([('\033[30minput_%d\033[0m' % ch, channel) for (ch, channel) in zip(channels, x_train)])
            net.fit(inputs, y_train)

        print(c['g'] + '-- Creating the test probability maps' + c['nc'])
        image_nii = load_nii(names[0, i])
        image = image_nii.get_data()
        for batch, centers in load_patch_batch(names[:, i], options['batch_size'], tuple(options['patch_size'])):
            if options['multi_channel']:
                y_pred = net.predict_proba(batch)
            else:
                batch = np.split(batch, n_channels, axis=1)
                inputs = dict([('\033[30minput_%d\033[0m' % ch, channel) for (ch, channel) in zip(channels, batch)])
                y_pred = net.predict_proba(inputs)
            [x, y, z] = np.stack(centers, axis=1)
            image[x, y, z] = y_pred[:, 1]

        image_nii.get_data()[:] = image
        name = mode_write + '.c' + str(i) + '.' + sufixes + '.nii.gz'
        path = '/'.join(names[0, i].rsplit('/')[:-1])
        image_nii.to_filename(os.path.join(path, name))


def patches_network_segmentation(options, mode):
    c = color_codes()
    image_sufix = get_sufix(
        options['use_flair'],
        options['use_pd'],
        options['use_t2'],
        options['use_gado'],
        options['use_t1']
    )
    size_sufix = '.'.join([str(length) for length in tuple(options['patch_size'])])
    sufixes = image_sufix + '.' + size_sufix
    mode_write = mode + '.mc' if options['multi_channel'] else mode + '.sc'

    print(c['g'] + 'Loading the data for the patch-based ' + c['b'] + mode + c['nc'])
    # Create the data
    (x, y, names) = load_patches(
        dir_name=options['folder'],
        use_flair=options['use_flair'],
        use_pd=options['use_pd'],
        use_t2=options['use_t2'],
        use_gado=options['use_gado'],
        use_t1=options['use_t1'],
        flair_name=options['flair'],
        pd_name=options['pd'],
        t2_name=options['t2'],
        gado_name=options['gado'],
        t1_name=options['t1'],
        mask_name=options['mask'],
        size=tuple(options['patch_size'])
    )

    print(c['g'] + 'Starting leave-one-out for the patch-based ' + c['b'] + mode + c['nc'])

    n_channels = x[0].shape[1]
    channels = range(0, n_channels)

    for x_train, y_train, i in leave_one_out(x, y):
        print('Running patient ' + c['c'] + names[0, i].rsplit('/')[-2] + c['nc'])
        seed = np.random.randint(np.iinfo(np.int32).max)
        print('-- Permuting the data')
        np.random.seed(seed)
        x_train = np.random.permutation(np.concatenate(x_train).astype(dtype=np.float32))
        print('-- Permuting the labels')
        np.random.seed(seed)
        y_train = np.random.permutation(np.concatenate(y_train).astype(dtype=np.int32))
        y_train = y_train.reshape([y_train.shape[0], -1])
        print('-- Training vector shape = (' + ','.join([str(length) for length in x_train.shape]) + ')')
        print('-- Training labels shape = (' + ','.join([str(length) for length in y_train.shape]) + ')')

        print(c['g'] + '-- Creating the ' + c['b'] + 'patch-based ' + c['b'] + mode + c['nc'])

        # Train the net and save it
        net_name = os.path.join(
            os.path.split(names[0, i])[0], 'patches_' + mode + '.c' + str(i) + '.' + sufixes
        )
        net_types = {
            'unet': create_unet3d_seg_string,
            'unet-short': create_unet3d_shortcuts_seg_string
        }
        net = net_types[mode](
            ''.join(options['layers']),
            x_train.shape,
            options['convo_size'],
            options['pool_size'],
            options['number_filters'],
            options['patience'],
            options['multi_channel'],
            net_name
        )

        print(c['g'] + '-- Training the ' + c['b'] + 'patch-based ' + c['b'] + mode + c['nc'])
        # We try to get the last weights to keep improving the net over and over
        try:
            net.load_params_from(net_name + 'model_weights.pkl')
        except IOError:
            pass

        if options['multi_channel']:
            net.fit(x_train, y_train)
        else:
            x_train = np.split(x_train, n_channels, axis=1)
            inputs = dict(
                [('\033[30minput_%d\033[0m' % ch, channel) for (ch, channel) in zip(channels, x_train)])
            net.fit(inputs, y_train)

        print(c['g'] + '-- Creating the test probability maps' + c['nc'])
        image_nii = load_nii(names[0, i])
        image = np.zeros_like(image_nii.get_data())
        for batch, centers in load_patch_batch(names[:, i], options['batch_size'],
                                               tuple(options['patch_size'])):
            if options['multi_channel']:
                y_pred = net.predict_proba(batch)
            else:
                batch = np.split(batch, n_channels, axis=1)
                inputs = dict(
                    [('\033[30minput_%d\033[0m' % ch, channel) for (ch, channel) in zip(channels, batch)])
                y_pred = net.predict_proba(inputs)

            image += sum_patches_to_image(y_pred, centers, image)

        image_nii.get_data()[:] = image
        name = mode_write + '.c' + str(i) + '.' + sufixes + '.nii.gz'
        path = '/'.join(names[0, i].rsplit('/')[:-1])
        image_nii.to_filename(os.path.join(path, name))


if __name__ == '__main__':
    main()
