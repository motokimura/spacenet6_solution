#!/usr/bin/env python3
# this script generates building footprint and boundary mask PNG files
# from given building polygon GeoJSON files
# see also notebooks/geojson_to_mask.ipynb

import solaris as sol
import os
from skimage import io
import geopandas as gpd
import numpy as np
from tqdm import tqdm


def check_filenames_validity(sar_image_filename, building_label_filename):
    # check if the filenames are valid
    assert sar_image_filename[:41] == 'SN6_Train_AOI_11_Rotterdam_SAR-Intensity_'
    assert sar_image_filename[-4:] == '.tif'

    assert building_label_filename[:37] == 'SN6_Train_AOI_11_Rotterdam_Buildings_'
    assert building_label_filename[-8:] == '.geojson'

    # check if both of sar image and building label cover the same tile
    assert sar_image_filename[41:-4] == building_label_filename[37:-8]


def get_roi_mask(sar_image):
    mask = sar_image.sum(axis=-1) > 0.0
    return mask


def combine_masks(footprint_mask, boundary_mask):
    h, w = footprint_mask.shape
    combined_mask = np.zeros(shape=(h, w), dtype=np.uint8)
    combined_mask[boundary_mask > 0] = 2  # 2 for boundary
    combined_mask[footprint_mask > 0] = 1  # 1 for footprint

    combined_mask_color = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    combined_mask_color[combined_mask == 1] = np.array([255, 0, 0])  # red for footprint
    combined_mask_color[combined_mask == 2] = np.array([0, 255, 0])  # green for boundary

    return (
        combined_mask, ## 0: background, 1: footprint, 2: boundary
        combined_mask_color  # black: background, red: footprint, green: boundary
    )


if __name__ == '__main__':
    # parameters
    data_dir = '/data/spacenet6/spacenet6/train/'
    out_dir = '/data/spacenet6/footprint_boundary_mask/'
    boundary_type = 'outer'  # 'outer' or 'inner'
    boundary_width_px = 10

    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels_color'), exist_ok=True)

    # SAR intensity
    sar_image_dir = os.path.join(data_dir, 'SAR-Intensity')
    sar_image_filenames = os.listdir(sar_image_dir)
    sar_image_filenames.sort()

    # building label
    building_label_dir = os.path.join(data_dir, 'Buildings')
    building_label_filenames = os.listdir(building_label_dir)
    building_label_filenames.sort()

    N = 3401
    assert len(building_label_filenames) == N
    assert len(sar_image_filenames) == N

    for i in tqdm(range(N)):
        sar_image_filename = sar_image_filenames[i]
        building_label_filename = building_label_filenames[i]
        check_filenames_validity(sar_image_filename, building_label_filename)

        # load sar image and compute roi mask from sar intensity values
        sar_image_path = os.path.join(sar_image_dir, sar_image_filename)
        sar_image = io.imread(sar_image_path)
        roi_mask = get_roi_mask(sar_image)

        # load building label
        building_label_path = os.path.join(building_label_dir, building_label_filename)

        # gen building footprint mask of the roi
        footprint_mask = sol.vector.mask.footprint_mask(
            df=building_label_path,
            reference_im=sar_image_path
        )
        footprint_mask[np.logical_not(roi_mask)] = 0

        # gen building boundary mask of the roi
        boundary_mask = sol.vector.mask.boundary_mask(
            footprint_mask,
            boundary_width=boundary_width_px,
            boundary_type=boundary_type
        )
        boundary_mask[np.logical_not(roi_mask)] = 0

        # combine footprint mask and boundary mask
        combined_mask, combined_mask_color = combine_masks(footprint_mask, boundary_mask)

        # save masks
        out_filename = f'{building_label_filename}.png'
        io.imsave(os.path.join(out_dir, 'labels', out_filename), combined_mask)
        io.imsave(os.path.join(out_dir, 'labels_color', out_filename), combined_mask_color)
