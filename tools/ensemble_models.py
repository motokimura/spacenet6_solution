#!/usr/bin/env python3
# this script conducts model ensemble by averaging 
# score arrays output by different multiple models

import argparse
import numpy as np
import os.path

from glob import glob
from skimage import io
from tqdm import tqdm

from _init_path import init_path
init_path()

from spacenet6_model.utils import experiment_subdir, get_roi_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_ids',
        help='experiment id of the models to ensemble',
        nargs='+',
        type=int,
        required=True
    )
    parser.add_argument(
        '--out_dir',
        help='output root directory',
        required=True
    )
    parser.add_argument(
        '--test_image_dir',
        help='directory containing spacenet6 test images',
        default='/data/spacenet6/spacenet6/test_public/SAR-Intensity/'
    )
    parser.add_argument(
        '--pred_root_dir',
        help='directory containing experiment folders',
        default='/predictions/'
    )
    parser.add_argument(
        '--pred_channel',
        help='number of channels of score arrays',
        type=int,
        default=2
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    sar_image_paths = glob(os.path.join(args.test_image_dir, '*.tif'))
    sar_image_paths.sort()
    N = len(args.exp_ids)

    os.makedirs(args.out_dir, exist_ok=False)

    for sar_image_path in tqdm(sar_image_paths):
        sar_image = io.imread(sar_image_path)
        roi_mask = get_roi_mask(sar_image)

        h, w = roi_mask.shape
        assert h == 900 and w == 900
        ensembled_score = np.zeros(shape=[args.pred_channel, h, w])

        sar_image_filename = os.path.basename(sar_image_path)
        array_filename, _ = os.path.splitext(sar_image_filename)
        array_filename = f'{array_filename}.npy'

        for exp_id in args.exp_ids:
            exp_subdir = experiment_subdir(exp_id)
            score_array = np.load(
                os.path.join(
                    args.pred_root_dir,
                    exp_subdir,
                    array_filename
                )
            )
            score_array[:, np.logical_not(roi_mask)] = 0
            assert score_array.min() >= 0 and score_array.max() <= 1
            ensembled_score += score_array

        ensembled_score = ensembled_score / N
        assert ensembled_score.min() >= 0 and ensembled_score.max() <= 1
        np.save(os.path.join(args.out_dir, array_filename), ensembled_score)
