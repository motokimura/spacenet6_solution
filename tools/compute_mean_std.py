#!/usr/bin/env python3
# this script computes mean and std values of given images
# forr each channel

import argparse
import numpy as np
import os
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='directory containing spacenet6 train dataset',
        default='/data/spacenet6/spacenet6/train/'
    )
    parser.add_argument(
        '--image_subdir',
        help='sub directory containing images under data_dir',
        default='SAR-Intensity'
    )
    parser.add_argument(
        '--out_dir',
        help='output root directory',
        default='/data/spacenet6/image_mean_std/'
    )
    return parser.parse_args()


def get_image_shape(image_subdir):
    if image_subdir == 'SAR-Intensity':
        image_width, image_height, image_channel = 900, 900, 4
    elif image_subdir == 'PS-MS':
        image_width, image_height, image_channel = 900, 900, 4
    elif image_subdir == 'PAN':
        image_width, image_height, image_channel = 900, 900, 1
    else:
        raise ValueError
    return (image_width, image_height, image_channel)


if __name__ == '__main__':
    args = parse_args()

    image_dir = os.path.join(args.data_dir, args.image_subdir)

    out_dir = os.path.join(args.out_dir, args.image_subdir)
    os.makedirs(out_dir, exist_ok=True)

    image_width, image_height, image_channel = \
        get_image_shape(args.image_subdir)

    if image_channel == 1:
        mean = np.zeros(shape=(image_height, image_width))
        std = np.zeros(shape=(image_height, image_width))
    else:
        mean = np.zeros(shape=(image_height, image_width, image_channel))
        std = np.zeros(shape=(image_height, image_width, image_channel))

    image_filenames = os.listdir(image_dir)
    N = len(image_filenames)
    assert N == 3401

    print('computing mean...')
    for image_filename in tqdm(image_filenames):
        image = io.imread(os.path.join(image_dir, image_filename))
        mean += image / N
    mean = np.mean(mean, axis=(0, 1))  # [image_channel,] or scaler if image_channel==1

    if image_channel == 1:
        mean_ = mean  # scaler
    else:
        mean_ = mean[None, None, :]  # [1, 1, image_channel]

    print('computing std...')
    for image_filename in tqdm(image_filenames):
        image = io.imread(os.path.join(image_dir, image_filename))
        std += (image - mean_) ** 2.0 / N
    std = np.mean(std, axis=(0, 1))  # [image_channel,] or scaler if image_channel==1
    std = np.sqrt(std)

    print(f'mean: {mean}')
    print(f'std: {std}')

    np.save(os.path.join(out_dir, 'mean.npy'), mean)
    np.save(os.path.join(out_dir, 'std.npy'), std)
