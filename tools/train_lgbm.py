#!/usr/bin/env python3

import argparse
import lightgbm as lgb
import os
import timeit

from _init_path import init_path
init_path()

from spacenet6_model.datasets.utils import read_orientation_file
from spacenet6_model.utils import (
    compute_features, evaluate_solution_csv, get_dataset, get_labels
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--truth_csvs',
        help='path to ground truth csv files',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--solution_csvs',
        help='path to solution csv files',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--image_dir',
        help='path directory containing SAR tif images',
        required=True
    )
    parser.add_argument(
        '--sar_orientation',
        help='path to SAR_orientations.txt',
        required=True
    )
    parser.add_argument(
        '--out_dir',
        help='path to directory to output models',
        required=True
    )
    return parser.parse_args()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parse_args()
    assert len(args.truth_csvs) == len(args.solution_csvs)
    N = len(args.truth_csvs)

    os.makedirs(args.out_dir, exist_ok=False)

    # road SAR_orientations.txt
    rotation_df = read_orientation_file(args.sar_orientation)

    # evaluate iou_score of each polygon of solution csv file
    solution_dfs = []
    for truth_csv, solution_csv in zip(args.truth_csvs, args.solution_csvs):
        df = evaluate_solution_csv(truth_csv, solution_csv)
        solution_dfs.append(df)

    # get features and label (iou_score) of each polygon to train LGBM
    features, labels = [], []
    for i, df in enumerate(solution_dfs):
        print(f'processing split #{i}...')
        feature = compute_features(df, args.image_dir, rotation_df)
        label = get_labels(df)
        features.append(feature)
        labels.append(label)

    # train LGBM in N fold
    for val_idx in range(N):
        # val dataset
        val_idxs = [val_idx,]
        X_val, y_val = get_dataset(features, labels, df_idxs=val_idxs)
        # train dataset
        train_idxs = list(range(N))
        train_idxs.remove(val_idx)
        X_train, y_train = get_dataset(features, labels, df_idxs=train_idxs)

        # train
        train_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_val, y_val)

        params = {
            'objective': 'regression',
            'metric': 'l2',
            'num_leaves': 28,
            'learning_rate': 0.07,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'verbosity': 0,
            'num_threads': 16
        }

        gbm = lgb.train(
            params,
            train_data,
            valid_sets=val_data,
            num_boost_round=2000,
            early_stopping_rounds=5,
            verbose_eval=True,
        )

        gbm.save_model(os.path.join(
            args.out_dir,
            f'gbm_model_{val_idx}.txt'
        ))

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
