import os
import time
import re
from operator import itemgetter
import numpy as np
from scipy import ndimage as nd
from nibabel import load as load_nii
from nibabel import save as save_nii
from nibabel import Nifti1Image as NiftiImage
from data_manipulation.generate_features import get_mask_voxels, get_patches, get_patches2_5d
from utils import color_codes
from itertools import izip


def sum_patch_to_image(patch, center, image):
    patch_size = patch.shape
    patch_half = tuple([idx / 2 for idx in patch_size])
    indices = [slice(c_idx - p_idx, c_idx + p_idx + 1) for (c_idx, p_idx) in izip(center, patch_half)]
    image[indices] += patch
    return image


def sum_patches_to_image(patches, centers, image):
    return np.sum(map(lambda p, c: sum_patch_to_image(p, c, image), patches, centers))


def set_patches(image, centers, patches, patch_size=(15, 15, 15)):
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = all([patch_size == patch.shape for patch in patches])
    if list_of_tuples and sizes_match:
        patch_half = tuple([idx/2 for idx in patch_size])
        slices = [
            [slice(c_idx - p_idx, c_idx + p_idx + 1) for (c_idx, p_idx) in izip(center, patch_half)]
            for center in centers
            ]
        for sl, patch in izip(slices, patches):
            image[sl] = patch
    return patches


def save_nifti(image, name):
    # Reshape the image to the original image's size and save it as nifti
    # In this case, we add "_reshape" no the original image's name to
    # remark that it comes from an autoencoder
    nifti = NiftiImage(image, affine=np.eye(4))
    print '\033[32;1mSaving\033[0;32m to \033[0m' + name + '\033[32m ...\033[0m'
    save_nii(nifti, name)
    # Return it too, just in case
    return nifti


def reshape_to_nifti(image, original_name):
    # Open the original nifti
    original = load_nii(original_name).get_data()
    # Reshape the image and save it
    reshaped = nd.zoom(
        image,
        [
            float(original.shape[0]) / image.shape[0],
            float(original.shape[1]) / image.shape[1],
            float(original.shape[2]) / image.shape[2]
        ]
    )
    reshaped *= original.std()
    reshaped += original.mean()
    reshaped_nii = NiftiImage(reshaped, affine=np.eye(4))

    return reshaped_nii


def reshape_save_nifti(image, original_name):
    # Reshape the image to the original image's size and save it as nifti
    # In this case, we add "_reshape" no the original image's name to
    # remark that it comes from an autoencoder
    reshaped_nii = reshape_to_nifti(image, original_name)
    new_name = re.search(r'(.+?)\.nii.*|\.+', original_name).groups()[0] + '_reshaped.nii.gz'
    print '\033[32;1mSaving\033[0;32m to \033[0m' + new_name + '\033[32m ...\033[0m'
    save_nii(reshaped_nii, new_name)
    # Return it too, just in case
    return reshaped_nii


def reshape_save_nifti_to_dir(image, original_name):
    # Reshape the image to the original image's size and save it as nifti
    # In this case, we save the probability map to the directory of the
    # original image with the name "unet_prob.nii.gz
    reshaped_nii = reshape_to_nifti(image, original_name)
    new_name = os.path.join(original_name[:original_name.rfind('/')], 'unet_prob.nii.gz')
    print '\033[32;1mSaving\033[0;32m to \033[0m' + new_name + '\033[32m ...\033[0m'
    save_nii(reshaped_nii, new_name)
    # Return it too, just in case
    return reshaped_nii


def load_thresholded_images(name, dir_name, threshold=2.0, datatype=np.float32):
    patients = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    image_names = [os.path.join(dir_name, patient, name) for patient in patients]
    images = [load_nii(image_name).get_data() for image_name in image_names]
    rois = [image.astype(dtype=datatype) > threshold for image in images]
    return rois


def load_masks(mask_names):
    for image_name in mask_names:
        yield load_nii(image_name).get_data().astype(dtype=np.bool)


def threshold_image_list(images, threshold, masks=None):
    return [im * m > threshold for im, m in izip(images, masks)] if masks else [im > threshold for im in images]


def load_thresholded_images_by_name(image_names, threshold=2.0):
    images = [load_nii(image_name).get_data() for image_name in image_names]
    return threshold_image_list(images, threshold)


def load_thresholded_norm_images(name, dir_name, threshold=2.0):
    patients = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    image_names = [os.path.join(dir_name, patient, name) for patient in patients]
    return threshold_image_list(norm_image_generator(image_names), threshold)


def load_thresholded_norm_images_by_name(image_names, mask_names=None, threshold=2.0):
    masks = [load_nii(mask).get_data() for mask in mask_names] if mask_names else None
    return threshold_image_list(norm_image_generator(image_names), threshold, masks)


def load_image_vectors(name, dir_name, min_shape, datatype=np.float32):
    # Get the names of the images and load them
    patients = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    image_names = [os.path.join(dir_name, patient, name) for patient in patients]
    images = [load_nii(image_name).get_data() for image_name in image_names]
    # Reshape everything to have data of homogenous size (important for training)
    # Also, normalize the data
    if min_shape is None:
        min_shape = min([im.shape for im in images])
    data = np.asarray(
        [nd.zoom((im - im.mean()) / im.std(),
                 [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                  float(min_shape[2]) / im.shape[2]]) for im in images]
    )

    return data.astype(datatype), image_names


def norm_image_generator(image_names):
    for name in image_names:
        im = load_nii(name).get_data()
        yield (im - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std()


def norm_defo_generator(image_names):
    for name in image_names:
        im = load_nii(name).get_data()
        yield im / np.linalg.norm(im, axis=4).std()


def load_patch_batch_percent(
        image_names,
        batch_size,
        size,
        defo_size=None,
        d_names=None,
        mask=None,
        datatype=np.float32
):
    images = [load_nii(name).get_data() for name in image_names]
    defos = [load_nii(name).get_data() for name in d_names] if d_names is not None else []
    images_norm = [(im - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    defos_norm = [im / np.linalg.norm(im, axis=4).std() for im in defos]
    mask = images[0].astype(np.bool) if mask is None else mask.astype(np.bool)
    lesion_centers = get_mask_voxels(mask)
    n_centers = len(lesion_centers)
    for i in range(0, n_centers, batch_size):
        centers = lesion_centers[i:i + batch_size]
        x = get_image_patches(images_norm, centers, size).astype(dtype=datatype)
        d = get_defo_patches(defos_norm, centers, size=defo_size) if defos else []
        patches = (x, d) if defos else x
        yield patches, centers, (100.0 * min((i + batch_size),  n_centers)) / n_centers


def subsample(center_list, sizes, random_state):
    np.random.seed(random_state)
    indices = [np.random.permutation(range(0, len(centers))).tolist()[:size]
               for centers, size in izip(center_list, sizes)]
    return [itemgetter(*idx)(centers) if idx else [] for centers, idx in izip(center_list, indices)]


def get_defo_patches(defos, centers, size=(5, 5, 5)):
    ds_xyz = [np.split(d, 3, axis=4) for d in defos]
    defo_patches = [
        np.stack(
            [get_patches(np.squeeze(d), centers, size) for d in d_xyz],
            axis=1
        ) for d_xyz in ds_xyz
    ]
    patches = np.stack(defo_patches, axis=1)
    return patches


def get_image_patches(image_list, centers, size):
    patches = np.stack(
        [np.array(get_patches(image, centers, size)) for image in image_list],
        axis=1,
    ) if len(size) == 3 else np.array([np.stack(get_patches2_5d(image, centers, size)) for image in image_list])
    return patches


def get_list_of_patches(image_list, center_list, size):
    patches = [
        get_patches(image, centers, size) for image, centers in izip(image_list, center_list) if centers
        ] if len(size) == 3 else [
        np.stack(get_patches2_5d(image, centers, size)) for image, centers in izip(image_list, center_list) if centers
        ]
    return patches


def get_centers_from_masks(positive_masks, negative_masks, balanced=True, random_state=42):
    positive_centers = [get_mask_voxels(mask) for mask in positive_masks]
    negative_centers = [get_mask_voxels(mask) for mask in negative_masks]
    if balanced:
        positive_voxels = [len(positives) for positives in positive_centers]
        negative_centers = list(subsample(negative_centers, positive_voxels, random_state))

    return positive_centers, negative_centers


def get_norm_patch_vectors(image_names, positive_masks, negative_masks, size, balanced=True, random_state=42):
    # Get all the centers for each image
    c = color_codes()
    print(c['lgy'] + '                ' + image_names[0].rsplit('/')[-1] + c['nc'])

    # Get all the patches for each image
    positive_centers, negative_centers = get_centers_from_masks(positive_masks, negative_masks, balanced, random_state)
    return get_patch_vectors(norm_image_generator(image_names), positive_centers, negative_centers, size)


def get_defo_patch_vectors(image_names, masks, size=(5, 5, 5), balanced=True, random_state=42):
    # Get all the centers for each image
    c = color_codes()
    print(c['lgy'] + '                ' + image_names[0].rsplit('/')[-1] + c['nc'])
    defo = norm_defo_generator(image_names)

    # We divide the 4D deformation image into 3D images for each component (x, y, z)
    defo_xyz = [np.squeeze(d) for d in [np.split(d, 3, axis=4) for d in defo]]

    positive_masks, negative_masks = masks

    positive_centers, negative_centers = get_centers_from_masks(positive_masks, negative_masks, balanced, random_state)

    patches = np.stack(
        [np.concatenate(get_patch_vectors(list(d), positive_centers, negative_centers, size))
         for d in izip(*defo_xyz)],
        axis=1
    )

    # Get all the patches for each image
    return patches


def get_patch_vectors(images, positive_centers, negative_centers, size):
    centers = [p + list(n) for p, n in izip(positive_centers, negative_centers)]
    patches = get_list_of_patches(images, centers, size)

    # Return the patch vectors
    data = patches if len(size) == 3 else [np.swapaxes(p, 0, 1) for p in izip(patches)]
    return data


def load_patch_vectors(name, mask_name, dir_name, size, rois=None, random_state=42):
    # Get the names of the images and load them
    patients = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    image_names = [os.path.join(dir_name, patient, name) for patient in patients]
    # Create the masks
    brain_masks = rois if rois else load_masks(image_names)
    mask_names = [os.path.join(dir_name, patient, mask_name) for patient in patients]
    lesion_masks = load_masks(mask_names)
    nolesion_masks = [np.logical_and(np.logical_not(lesion), brain) for lesion, brain in
                      izip(lesion_masks, brain_masks)]

    # Get all the patches for each image
    # Get all the centers for each image
    positive_centers = [get_mask_voxels(mask) for mask in lesion_masks]
    negative_centers = [get_mask_voxels(mask) for mask in nolesion_masks]
    positive_voxels = [len(positives) for positives in positive_centers]
    nolesion_small = subsample(negative_centers, positive_voxels, random_state)

    # Get all the patches for each image
    images = norm_image_generator(image_names)
    positive_patches = get_list_of_patches(images, positive_centers, size)
    images = norm_image_generator(image_names)
    negative_patches = get_list_of_patches(images, nolesion_small, size)

    # Prepare the mask patches for training
    positive_mask_patches = get_list_of_patches(lesion_masks, positive_centers, size)
    negative_mask_patches = get_list_of_patches(nolesion_masks, nolesion_small, size)

    # Return the patch vectors
    data = [np.concatenate([p1, p2]) for p1, p2 in izip(positive_patches, negative_patches)]
    masks = [np.concatenate([p1, p2]) for p1, p2 in izip(positive_mask_patches, negative_mask_patches)]

    return data, masks, image_names


def get_cnn_rois(names, mask_names, roi_names=None, pr_names=None, th=1.0, balanced=True):
    rois = load_thresholded_norm_images_by_name(
        names[0, :],
        threshold=th,
        mask_names=roi_names
    ) if roi_names is not None else load_masks(names)
    if pr_names is not None:
        pr_maps = [load_nii(name).get_data() * roi for name, roi in izip(pr_names, rois)]
        if balanced:
            idx_sorted_maps = [np.argsort(pr_map * np.logical_not(lesion_mask), axis=None)
                               for pr_map, lesion_mask in izip(pr_maps, load_masks(mask_names))]
            rois_n = [idx.reshape(lesion_mask.shape) > (idx.shape[0] - np.sum(lesion_mask) - 1)
                      for idx, lesion_mask in izip(idx_sorted_maps, load_masks(mask_names))]
        else:
            rois_n = [np.logical_and(np.logical_not(lesion_mask), pr_map > 0.5)
                      for pr_map, lesion_mask in izip(pr_maps, load_masks(mask_names))]
    else:
        rois_n = [np.logical_and(np.logical_not(lesion), brain)
                  for lesion, brain in izip(load_masks(mask_names), rois)]

    rois_p = list(load_masks(mask_names))
    return rois_p, rois_n


def load_and_stack(names, rois, patch_size, balanced=True, random_state=42):
    rois_p, rois_n = rois

    images_loaded = [
        get_norm_patch_vectors(
            names_i,
            rois_p,
            rois_n,
            patch_size,
            balanced=balanced,
            random_state=random_state
        ) for names_i in names]

    x_train = [np.stack(images, axis=1) for images in izip(*images_loaded)]
    y_train = [
        np.concatenate([np.ones(x.shape[0] / 2), np.zeros(x.shape[0] / 2)])
        for x in x_train
        ] if balanced else [
        np.concatenate([np.ones(sum(roi_p.flatten())), np.zeros(sum(roi_n.flatten()))])
        for roi_p, roi_n in izip(rois_p, rois_n)
        ]

    return x_train, y_train, (rois_p, rois_n)


def permute(x, seed, datatype=np.float32):
    c = color_codes()
    print(c['g'] + '                Vector shape ='
          ' (' + ','.join([c['bg'] + str(length) + c['nc'] + c['g'] for length in x.shape]) + ')' + c['nc'])
    np.random.seed(seed)
    x_permuted = np.random.permutation(x.astype(dtype=datatype))

    return x_permuted


def load_lesion_cnn_data(
        names,
        mask_names,
        roi_names,
        init_pr_names=None,
        pr_names=None,
        defo_names=None,
        patch_size=(11, 11, 11),
        defo_size=(5, 5, 5),
        balanced=True,
        random_state=42,
):
    seed = time.clock() if not random_state else random_state
    pr_names = names[0, :] if pr_names is None else pr_names
    rois = get_cnn_rois(names, mask_names, roi_names=roi_names, pr_names=pr_names, balanced=balanced)
    if init_pr_names is not None:
        rois_p, i1rois_n = rois
        _, i2rois_n = get_cnn_rois(
            names,
            mask_names,
            roi_names=roi_names,
            pr_names=pr_names,
            balanced=balanced
        )
        rois_n = [np.logical_or(ri1_n, ri2_n) for ri1_n, ri2_n in zip(i1rois_n, i2rois_n)]
        rois = (rois_p, rois_n)

    print('                Loading image data and labels vector')
    x_train, y_train, rois = load_and_stack(names, rois, patch_size, balanced=balanced, random_state=seed)
    x_train = np.concatenate(x_train)
    x_train = permute(x_train, seed)
    y_train = np.concatenate(y_train)
    y_train = permute(y_train, seed, datatype=np.int32)
    if defo_names is not None:
        print('                Creating deformation vector')
        defo_train = np.stack(
            [get_defo_patch_vectors(names_i, rois, size=defo_size, balanced=balanced, random_state=seed)
             for names_i in defo_names],
            axis=1
        )
        defo_train = permute(defo_train, seed)

        x_train = (x_train, defo_train)

    return x_train, y_train


def load_register_data(names, image_size, seed):
    print('                Creating data vector')
    images = [norm_image_generator(n) for n in names]
    images_loaded = [
        np.stack([nd.interpolation.zoom(im, [A/(1.0*B) for A, B in izip(image_size, im.shape)]) for im in gen])
        for gen in images]
    x_train = np.stack(images_loaded)
    x_train = np.concatenate([x_train, np.stack([x_train[:, 1, :, :, :], x_train[:, 0, :, :, :]], axis=1)])
    y_train = x_train[:, 1, :, :, :].reshape(x_train.shape[0], -1)
    print('                Permuting the data')
    np.random.seed(seed)
    x_train = np.random.permutation(x_train.astype(dtype=np.float32))
    print('                Permuting the labels')
    np.random.seed(seed)
    y_train = np.random.permutation(y_train.astype(dtype=np.float32))

    return x_train, y_train


def load_patches(
        dir_name,
        mask_name,
        flair_name,
        pd_name,
        t2_name,
        gado_name,
        t1_name,
        use_flair,
        use_pd,
        use_t2,
        use_gado,
        use_t1,
        size,
        roi_name=None
):
    # Setting up the lists for all images
    flair, flair_names = None, None
    pd, pd_names = None, None
    t2, t2_names = None, None
    t1, t1_names = None, None
    gado, gado_names = None, None
    y = None

    random_state = np.random.randint(1)

    # We load the image modalities for each patient according to the parameters
    rois = load_thresholded_images(roi_name, dir_name, threshold=0.5) if roi_name \
        else load_thresholded_norm_images(flair_name, dir_name, threshold=1)
    if use_flair:
        print 'Loading ' + flair_name + ' images'
        flair, y, flair_names = load_patch_vectors(flair_name, mask_name, dir_name, size, rois, random_state)
    if use_pd:
        print 'Loading ' + pd_name + ' images'
        pd, y, pd_names = load_patch_vectors(pd_name, mask_name, dir_name, size, rois, random_state)
    if use_t2:
        print 'Loading ' + t2_name + ' images'
        t2, y, t2_names = load_patch_vectors(t2_name, mask_name, dir_name, size, rois, random_state)
    if use_t1:
        print 'Loading ' + t1_name + ' images'
        t1, y, t1_names = load_patch_vectors(t1_name, mask_name, dir_name, size, rois, random_state)
    if use_gado:
        print 'Loading ' + gado_name + ' images'
        gado, y, gado_names = load_patch_vectors(gado_name, mask_name, dir_name, size, rois, random_state)

    print 'Creating data vector'
    data = [images for images in [flair, pd, t2, gado, t1] if images is not None]
    x = [np.stack(images, axis=1) for images in izip(*data)]
    image_names = np.stack([name for name in [
        flair_names,
        pd_names,
        t2_names,
        gado_names,
        t1_names
    ] if name is not None])

    return x, y, image_names
