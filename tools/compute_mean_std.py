#!/usr/bin/env python3
# this script computes mean and std values of given images
# in each channel independently

import numpy as np
import os
from skimage import io
from tqdm import tqdm


if __name__ == '__main__':
    # parameters
    image_dir = '../data/spacenet6/train/SAR-Intensity/'
    out_dir = '../data/image_mean_std/SAR-Intensiry/'

    os.makedirs(out_dir, exist_ok=True)

    mean = np.zeros(shape=(900, 900, 4))
    std = np.zeros(shape=(900, 900, 4))

    image_filenames = os.listdir(image_dir)
    N = len(image_filenames)
    assert N == 3401

    print('computing mean...')
    for image_filename in tqdm(image_filenames):
        image = io.imread(os.path.join(image_dir, image_filename))
        mean += image / N
    mean = np.mean(mean, axis=(0, 1))  # [4,]

    print('computing std...')
    for image_filename in tqdm(image_filenames):
        image = io.imread(os.path.join(image_dir, image_filename))
        std += (image - mean[None, None, :]) ** 2.0 / N
    std = np.mean(std, axis=(0, 1))  # [4,]
    std = np.sqrt(std)

    print(f'mean: {mean}')
    print(f'std: {std}')

    np.save(os.path.join(out_dir, 'mean.npy'), mean)
    np.save(os.path.join(out_dir, 'std.py'), std)
