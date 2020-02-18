#!/usr/bin/env python3
# this script computes mean and std values of given images
# forr each channel

import numpy as np
import os
from skimage import io
from tqdm import tqdm


if __name__ == '__main__':
    # parameters
    data_dir = '/data/spacenet6/train/'
    out_base_dir = '/data/image_mean_std/'
    # --SAR
    image_subdir = 'SAR-Intensity'
    image_width, image_height, image_channel = 900, 900, 4
    # --PAN
    #image_subdir = 'PAN'
    #image_width, image_height, image_channel = 900, 900, 1

    image_dir = os.path.join(data_dir, image_subdir)

    out_dir = os.path.join(out_base_dir, image_subdir)
    os.makedirs(out_dir, exist_ok=True)

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
