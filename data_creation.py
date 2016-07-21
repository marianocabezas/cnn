import os
import re
import gc
import numpy as np
from scipy import ndimage as nd
from nibabel import load as load_nii
from nibabel import save as save_nii
from nibabel import Nifti1Image as NiftiImage
from math import floor
from data_manipulation.generate_features import get_mask_voxels, get_patches
from operator import itemgetter


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


def sum_patch_to_image(patch, center, image):
    patch_size = patch.shape
    patch_half = tuple([idx / 2 for idx in patch_size])
    indices = [slice(c_idx - p_idx, c_idx + p_idx + 1) for (c_idx, p_idx) in zip(center, patch_half)]
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
            [slice(c_idx - p_idx, c_idx + p_idx + 1) for (c_idx, p_idx) in zip(center, patch_half)]
            for center in centers
            ]
        for sl, patch in zip(slices, patches):
            image[sl] = patch
    return patches


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


def load_thresholded_images_by_name(image_names, threshold=2.0, datatype=np.float32):
    images = [load_nii(image_name).get_data() for image_name in image_names]
    rois = [image.astype(dtype=datatype) > threshold for image in images]
    return rois


def load_thresholded_norm_images(name, dir_name, threshold=2.0, datatype=np.float32):
    patients = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    image_names = [os.path.join(dir_name, patient, name) for patient in patients]
    images = [load_nii(image_name).get_data() for image_name in image_names]
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    rois = [image > threshold for image in images_norm]
    return rois


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


def load_patch_batch(image_names, batch_size, size, datatype=np.float32):
    images = [load_nii(name).get_data() for name in image_names]
    images_norm = [(im - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    lesion_centers = get_mask_voxels(images[0].astype(np.bool))
    for i in range(0, len(lesion_centers), batch_size):
        centers = lesion_centers[i:i+batch_size]
        yield np.stack(
            [np.array(get_patches(image, centers, size)).astype(datatype) for image in images_norm], axis=1
        ), centers


def load_patch_batch_percent(image_names, batch_size, size, datatype=np.float32):
    images = [load_nii(name).get_data() for name in image_names]
    images_norm = [(im - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    lesion_centers = get_mask_voxels(images[0].astype(np.bool))
    n_centers = len(lesion_centers)
    for i in range(0, n_centers, batch_size):
        centers = lesion_centers[i:i + batch_size]
        yield np.stack(
            [np.array(get_patches(image, centers, size)).astype(datatype) for image in images_norm], axis=1
        ), centers, (100.0 * min((i + batch_size),  n_centers)) / n_centers


def load_patch_vectors(name, mask_name, dir_name, size, rois=None, random_state=42, datatype=np.float32):
    # Get the names of the images and load them
    patients = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    image_names = [os.path.join(dir_name, patient, name) for patient in patients]
    images = [load_nii(name).get_data() for name in image_names]
    # Normalize the images
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    # Create the masks
    brain_masks = rois if rois else [image.astype(dtype=np.bool) for image in images]
    mask_names = [os.path.join(dir_name, patient, mask_name) for patient in patients]
    lesion_masks = [load_nii(name).get_data().astype(dtype=np.bool) for name in mask_names]
    nolesion_masks = [np.logical_and(np.logical_not(lesion), brain) for lesion, brain in zip(lesion_masks, brain_masks)]

    # Get all the patches for each image
    lesion_centers = [get_mask_voxels(mask) for mask in lesion_masks]
    nolesion_centers = [get_mask_voxels(mask) for mask in nolesion_masks]
    # FIX: nolesion_small should have the best indices
    np.random.seed(random_state)
    indices = [np.random.permutation(range(0, len(centers1))).tolist()[:len(centers2)]
               for centers1, centers2 in zip(nolesion_centers, lesion_centers)]
    nolesion_small = [itemgetter(*idx)(centers) for centers, idx in zip(nolesion_centers, indices)]
    lesion_patches = [np.array(get_patches(image, centers, size))
                      for image, centers in zip(images_norm, lesion_centers)]
    lesion_msk_patches = [np.array(get_patches(image, centers, size))
                          for image, centers in zip(lesion_masks, lesion_centers)]
    nolesion_patches = [np.array(get_patches(image, centers, size))
                        for image, centers in zip(images_norm, nolesion_small)]
    nolesion_msk_patches = [np.array(get_patches(image, centers, size))
                            for image, centers in zip(lesion_masks, nolesion_small)]

    data = [np.concatenate([p1, p2]) for p1, p2 in zip(lesion_patches, nolesion_patches)]
    masks = [np.concatenate([p1, p2]) for p1, p2 in zip(lesion_msk_patches, nolesion_msk_patches)]

    return data, masks, image_names


def load_patch_vectors_by_name(names, mask_names, size, rois=None, random_state=42, datatype=np.float32):
    images = [load_nii(name).get_data() for name in names]
    # Normalize the images
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    # Create the masks
    brain_masks = rois if rois else [image.astype(dtype=np.bool) for image in images]
    lesion_masks = [load_nii(name).get_data().astype(dtype=np.bool) for name in mask_names]
    nolesion_masks = [np.logical_and(np.logical_not(lesion), brain) for lesion, brain in zip(lesion_masks, brain_masks)]

    # Get all the patches for each image
    lesion_centers = [get_mask_voxels(mask) for mask in lesion_masks]
    nolesion_centers = [get_mask_voxels(mask) for mask in nolesion_masks]
    # FIX: nolesion_small should have the best indices
    np.random.seed(random_state)
    indices = [np.random.permutation(range(0, len(centers1))).tolist()[:len(centers2)]
               for centers1, centers2 in zip(nolesion_centers, lesion_centers)]
    nolesion_small = [itemgetter(*idx)(centers) for centers, idx in zip(nolesion_centers, indices)]
    lesion_patches = [np.array(get_patches(image, centers, size))
                      for image, centers in zip(images_norm, lesion_centers)]
    lesion_msk_patches = [np.array(get_patches(image, centers, size))
                          for image, centers in zip(lesion_masks, lesion_centers)]
    nolesion_patches = [np.array(get_patches(image, centers, size))
                        for image, centers in zip(images_norm, nolesion_small)]
    nolesion_msk_patches = [np.array(get_patches(image, centers, size))
                            for image, centers in zip(lesion_masks, nolesion_small)]

    data = [np.concatenate([p1, p2]) for p1, p2 in zip(lesion_patches, nolesion_patches)]
    masks = [np.concatenate([p1, p2]) for p1, p2 in zip(lesion_msk_patches, nolesion_msk_patches)]

    return data, masks


def load_patch_vectors_by_name_pr(names, mask_names, size, pr_maps, datatype=np.float32):
    images = [load_nii(name).get_data() for name in names]
    # Normalize the images
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    # Create the masks
    lesion_masks = [load_nii(name).get_data().astype(dtype=np.bool) for name in mask_names]
    idx_sorted_maps = [np.argsort(pr_map * np.logical_not(lesion_mask), axis=None)
                       for pr_map, lesion_mask in zip(pr_maps, lesion_masks)]
    nolesion_masks = [idx.reshape(lesion_mask.shape) > (idx.shape[0] - np.sum(lesion_mask) - 1)
                      for idx, lesion_mask in zip(idx_sorted_maps, lesion_masks)]

    # Get all the patches for each image
    lesion_centers = [get_mask_voxels(mask) for mask in lesion_masks]
    nolesion_centers = [get_mask_voxels(mask) for mask in nolesion_masks]
    lesion_patches = [np.array(get_patches(image, centers, size))
                      for image, centers in zip(images_norm, lesion_centers)]
    lesion_msk_patches = [np.array(get_patches(image, centers, size))
                          for image, centers in zip(lesion_masks, lesion_centers)]
    nolesion_patches = [np.array(get_patches(image, centers, size))
                        for image, centers in zip(images_norm, nolesion_centers)]
    nolesion_msk_patches = [np.array(get_patches(image, centers, size))
                            for image, centers in zip(lesion_masks, nolesion_centers)]

    data = [np.concatenate([p1, p2]) for p1, p2 in zip(lesion_patches, nolesion_patches)]
    masks = [np.concatenate([p1, p2]) for p1, p2 in zip(lesion_msk_patches, nolesion_msk_patches)]

    return data, masks


def get_sufix(use_flair, use_pd, use_t2, use_t1, use_gado):
    images_used = [use_flair, use_pd, use_t2, use_t1, use_gado]
    letters = ['fl', 'pd', 't2', 't1', 'gd']

    return '.'.join([letter for (letter, is_used) in zip(letters, images_used) if is_used])


def load_images(
        dir_name,
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
        min_shape
):

    image_sufix = get_sufix(use_flair, use_pd, use_t2, use_gado, use_t1)

    try:
        x = np.load(os.path.join(dir_name, 'image_vector_encoder.' + image_sufix + '.npy'))
        np.load(os.path.join(dir_name, 'image_names_encoder.' + image_sufix + '.npy'))
    except IOError:
        # Setting up the lists for all images
        flair, flair_names = None, None
        pd, pd_names = None, None
        t2, t2_names = None, None
        t1, t1_names = None, None
        gado, gado_names = None, None

        # We load the image modalities for each patient according to the parameters
        if use_flair:
            print 'Loading ' + flair_name + ' images'
            flair, flair_names = load_image_vectors(flair_name, dir_name, min_shape=min_shape)
            gc.collect()
        if use_pd:
            print 'Loading ' + pd_name + ' images'
            pd, pd_names = load_image_vectors(pd_name, dir_name, min_shape=min_shape)
            gc.collect()
        if use_t2:
            print 'Loading ' + t2_name + ' images'
            t2, t2_names = load_image_vectors(t2_name, dir_name, min_shape=min_shape)
            gc.collect()
        if use_t1:
            print 'Loading ' + t1_name + ' images'
            t1, t1_names = load_image_vectors(t1_name, dir_name, min_shape=min_shape)
            gc.collect()
        if use_gado:
            print 'Loading ' + gado_name + ' images'
            gado, gado_names = load_image_vectors(gado_name, dir_name, min_shape=min_shape)
            gc.collect()

        x = np.stack([data for data in [flair, pd, t2, gado, t1] if data is not None], axis=1)
        image_names = np.stack([name for name in [
                flair_names,
                pd_names,
                t2_names,
                gado_names,
                t1_names
        ] if name is not None])
        np.save(os.path.join(dir_name, 'image_vector_encoder.' + image_sufix + '.npy'), x)
        np.save(os.path.join(dir_name, 'image_names_encoder.' + image_sufix + '.npy'), image_names)

    return x


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
    x = [np.stack(images, axis=1) for images in zip(*data)]
    image_names = np.stack([name for name in [
        flair_names,
        pd_names,
        t2_names,
        gado_names,
        t1_names
    ] if name is not None])

    return x, y, image_names


def load_encoder_data(
        dir_name,
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
        test_size=0.25,
        random_state=None,
        min_shape=None
):

    x = load_images(
        dir_name,
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
        min_shape
    )
    y = np.reshape(x, [x.shape[0], -1])

    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def load_unet_data(
        dir_name,
        flair_name,
        pd_name,
        t2_name,
        gado_name,
        t1_name,
        mask_name,
        use_flair,
        use_pd,
        use_t2,
        use_gado,
        use_t1,
        test_size=0.25,
        random_state=None,
        min_shape=None
):
    image_sufix = get_sufix(use_flair, use_pd, use_t2, use_gado, use_t1)

    try:
        y = np.load(os.path.join(dir_name, 'mask_vector_unet.' + image_sufix + '.npy'))
    except IOError:
        patients = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
        masks = [load_nii(os.path.join(dir_name, patient, mask_name)).get_data() for patient in
                 patients]
        min_shape = min([im.shape for im in masks])
        y = np.asarray(
            [nd.zoom(im,
                     [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                      float(min_shape[2]) / im.shape[2]]) for im in masks]
        ).astype(np.uint8)
        np.save(os.path.join(dir_name, 'mask_vector_unet.' + image_sufix + '.npy'), y)

    x = load_images(
        dir_name,
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
        min_shape
    )

    return train_test_split(x, y, test_size=test_size, random_state=random_state)
