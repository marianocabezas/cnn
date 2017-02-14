import os
import sys
import argparse
from nibabel import load as load_nii
from data_manipulation import metrics


def main():
    # Parse command line options
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    group = parser.add_mutually_exclusive_group()

    folder_help = 'Folder with the files to evaluate. Remember to include a init_names.py for the evaluation pairs.'
    files_help = 'Pair of files to be compared. The first is the GT and the second the file you wnat to evaluate.'

    group.add_argument('-f', '--folder', help=folder_help)
    group.add_argument('--files', nargs=2, help=files_help)
    args = parser.parse_args()

    if args.folder:
        folder_name = args.folder
        sys.path = sys.path + [folder_name]
        from init_names import get_names_from_folder
        gt_names, all_names = get_names_from_folder(folder_name)

    elif args.files:
        folder_name = os.getcwd()
        gt_names = [args.files[0]]
        all_names = [[args.files[1]]]

    with open(os.path.join(folder_name, 'results.csv'), 'w') as f:
        for gt_name, names in zip(gt_names, all_names):
            print('\033[32mEvaluating with ground truth \033[32;1m' + gt_name + '\033[0m')

            gt_nii = load_nii(gt_name)
            gt = gt_nii.get_data()
            spacing = dict(gt_nii.header.items())['pixdim'][1:4]

            for name in names:
                name = ''.join(name)
                print('\033[32m-- vs \033[32;1m' + name + '\033[0m')
                lesion = load_nii(name).get_data()
                dist = metrics.average_surface_distance(gt, lesion, spacing)
                tpfv = metrics.tp_fraction_seg(gt, lesion)
                fpfv = metrics.fp_fraction_seg(gt, lesion)
                dscv = metrics.dsc_seg(gt, lesion)
                tpfl = metrics.tp_fraction_det(gt, lesion)
                fpfl = metrics.fp_fraction_det(gt, lesion)
                dscl = metrics.dsc_det(gt, lesion)
                tp = metrics.true_positive_det(lesion, gt)
                gt_d = metrics.num_regions(gt)
                lesion_s = metrics.num_voxels(lesion)
                gt_s = metrics.num_voxels(gt)
                measures = (gt_name, name, dist, tpfv, fpfv, dscv, tpfl, fpfl, dscl, tp, gt_d, lesion_s, gt_s)
                if args.folder:
                    f.write('%s;%s;%f;%f;%f;%f;%f;%f;%f;%d;%d;%d;%d\n' % measures)
                else:
                    print('%s;%s;%f;%f;%f;%f;%f;%f;%f;%d;%d;%d;%d\n' % measures)


if __name__ == '__main__':
    main()