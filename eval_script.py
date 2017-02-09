import os
import sys
import getopt
from nibabel import load as load_nii
import itertools
from data_manipulation import metrics


def main():
    folder_name = '/home/mariano/DATA/LST/'

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)
    # process options
    for o, a in opts:
        if o in ('-h', '--help'):
            print('Help text')
            sys.exit(0)
        elif o in ('-f', '--folder'):
            folder_name = a

    patients = [f for f in sorted(os.listdir(folder_name)) if os.path.isdir(os.path.join(folder_name, f))]
    log_file = '%sresults.csv' % folder_name

    p_sizes = [9, 11]
    c_sizes = [3]
    d_sizes = [64, 128, 256]
    iters = ['i1', 'i2', 'f']
    iters_long = ['iter1', 'iter2', 'final']
    params = [p_sizes, c_sizes, d_sizes, iters]
    combos = ['p%d.c%d.d%d.%s.nii.gz' % p for p in list(itertools.product(*params))]
    output_combos = [pre + c for pre in ['joint_', 'joint_demons_', 'deep.'] for c in combos]

    with open(log_file, 'w') as f:
        for p in patients:
            print('\033[32mEvaluating patient \033[32;1m' + p + '\033[0m')
            gt_name = os.path.join(folder_name, p, 'gt_mask.nii')
            lga_name = os.path.join(folder_name, p, 'long_lga_mask.nii')
            lpa_name = os.path.join(folder_name, p, 'long_lpa_mask.nii')
            auto_name = os.path.join(folder_name, p, 'auto_mask.nii')
            onur_name = os.path.join(folder_name, p, 'joint_onur_positive_activity.nii.gz')
            greenspan_name = os.path.join(folder_name, p, 't' + p + '.greenspan.final.nii.gz')
            long_names = [os.path.join(
                folder_name, p, 't' + p + '.flair.pd.t2.p9.c2c2.n32.d256.pad_valid.' + i + '.nii.gz'
            ) for i in iters_long]
            defo_names = [os.path.join(
                folder_name, p, 't' + p + '.d.flair.pd.t2.p9.c2c2.n32.d256.pad_valid.' + i + '.nii.gz'
            ) for i in iters_long]
            deep_names = [os.path.join(folder_name, p, c) for c in output_combos]

            names = [auto_name, onur_name, lga_name, lpa_name] + deep_names + long_names + defo_names + [greenspan_name]

            gt_nii = load_nii(gt_name)
            gt = gt_nii.get_data()
            spacing = dict(gt_nii.header.items())['pixdim'][1:4]

            for name in names:
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
                measures = (p, name, dist, tpfv, fpfv, dscv, tpfl, fpfl, dscl, tp, gt_d, lesion_s, gt_s)
                f.write('%s;%s;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f\n' % measures)


if __name__ == '__main__':
    main()