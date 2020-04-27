#!/usr/bin/env python3

import argparse
import lightgbm as lgb
import pandas as pd
import timeit

from _init_path import init_path
init_path()

from spacenet6_model.datasets.utils import read_orientation_file
from spacenet6_model.utils import (
    compute_features, get_dataset, get_lgmb_prediction, image_has_no_polygon
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--solution_csv',
        help='path to solution csv',
        required=True
    )
    parser.add_argument(
        '--models',
        help='path to LGBM model files',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--out',
        help='path to output solution csv',
        required=True
    )
    parser.add_argument(
        '--iou_thresh',
        help='threshold of iou_score (0, 1)',
        type=float,
        default=0.15
    )
    parser.add_argument(
        '--image_dir',
        help='path directory containing SAR tif images',
        default='/data/spacenet6/spacenet6/test_public/AOI_11_Rotterdam/SAR-Intensity/'
    )
    parser.add_argument(
        '--sar_orientation',
        help='path to SAR_orientations.txt',
        default='/data/spacenet6/spacenet6/train/AOI_11_Rotterdam/SummaryData/SAR_orientations.txt'
    )
    return parser.parse_args()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parse_args()

    # road SAR_orientations.txt
    rotation_df = read_orientation_file(args.sar_orientation)

    # load models
    gbm_models = [
        lgb.Booster(model_file=path) for path in args.models
    ]

    # load solution csv
    solution_df = pd.read_csv(args.solution_csv)

    # prediction with ensemble
    dfs = []
    for image_id in solution_df['ImageId'].unique():
        image_df = solution_df[solution_df['ImageId'] == image_id]
    
        if image_has_no_polygon(image_df):
            dfs.append(image_df.copy(deep=True))
            continue

        x = compute_features(image_df, args.image_dir, rotation_df, is_train=False)
        y = get_lgmb_prediction(gbm_models, x)
        assert len(image_df) == len(y)

        tp_mask = y >= args.iou_thresh
        if tp_mask.sum() == 0:
            tmp = pd.DataFrame({
                'ImageId': [image_id,],
                'PolygonWKT_Pix': ['POLYGON EMPTY',],
                'Confidence': [1,]
            })
            dfs.append(tmp)
            continue

        dfs.append(image_df[tp_mask].copy(deep=True))

    result = pd.concat(dfs)
    result.to_csv(args.out, index=False)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
