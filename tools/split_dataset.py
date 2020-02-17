#!/usr/bin/env python3
# this script splits train dataset into train and val splits

import numpy as np
import os

def check_filename_validity(
    sar_image_filename,
    ms_image_filename,
    pan_image_filename,
    mask_filename
):
    # check if the filenames are valid
    assert sar_image_filename[:41] == 'SN6_Train_AOI_11_Rotterdam_SAR-Intensity_'
    assert sar_image_filename[-4:] == '.tif'

    assert ms_image_filename[:33] == 'SN6_Train_AOI_11_Rotterdam_PS-MS_'
    assert ms_image_filename[-4:] == '.tif'

    assert pan_image_filename[:31] == 'SN6_Train_AOI_11_Rotterdam_PAN_'
    assert pan_image_filename[-4:] == '.tif'

    assert mask_filename[:37] == 'SN6_Train_AOI_11_Rotterdam_Buildings_'
    assert mask_filename[-12:] == '.geojson.png'

    identity = sar_image_filename[41:-4]
    assert ms_image_filename[33:-4] == identity
    assert pan_image_filename[31:-4] == identity
    assert mask_filename[37:-12] == identity


def dump_to_file(out_path, data_list):
    N = len(data_list)
    with open(out_path, 'w') as f:
        for i in range(N):
            data = data_list[i]
            sar_image = data['SAR']
            ms_image = data['PS-MS']
            pan_image = data['PAN']
            mask = data['MASK']
            f.write(f'{sar_image} {ms_image} {pan_image} {mask}\n')


if __name__ == '__main__':
    # parameters
    data_dir = '../data/spacenet6/train/'
    mask_dir = '../data/footprint_boundary_mask/labels/'
    out_dir = '../data/split/'
    train_val_split_ratio = (0.8, 0.2)
    seed = 0

    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    # SAR intensity
    sar_image_dir = os.path.join(data_dir, 'SAR-Intensity')
    sar_image_filenames = os.listdir(sar_image_dir)
    sar_image_filenames.sort()

    # Pansharpened Multi-Spectral
    ms_image_dir = os.path.join(data_dir, 'PS-MS')
    ms_image_filenames = os.listdir(ms_image_dir)
    ms_image_filenames.sort()

    # Panchromatic
    pan_image_dir = os.path.join(data_dir, 'PAN')
    pan_image_filenames = os.listdir(pan_image_dir)
    pan_image_filenames.sort()

    # building mask
    mask_filenames = os.listdir(mask_dir)
    mask_filenames.sort()

    N = 3401
    assert len(sar_image_filenames) == N
    assert len(ms_image_filenames) == N
    assert len(pan_image_filenames) == N
    assert len(mask_filenames) == N

    data_list = []

    for i in range(N):
        sar_image_filename = sar_image_filenames[i]
        ms_image_filename = ms_image_filenames[i]
        pan_image_filename =  pan_image_filenames[i]
        mask_filename = mask_filenames[i]

        check_filename_validity(
            sar_image_filename,
            ms_image_filename,
            pan_image_filename,
            mask_filename
        )

        data_list.append(
            {
                'SAR': sar_image_filename,
                'PS-MS': ms_image_filename,
                'PAN': pan_image_filename,
                'MASK': mask_filename
            }
        )

    np.random.shuffle(data_list)

    ratio_train, ratio_val = train_val_split_ratio
    N_train = int(N * ratio_train / (ratio_train + ratio_val))
    train_list = data_list[:N_train]
    val_list = data_dir[N_train:]

    dump_to_file(os.path.join(out_dir, 'train.txt'), train_list)
    dump_to_file(os.path.join(out_dir, 'val.txt'), val_list)
