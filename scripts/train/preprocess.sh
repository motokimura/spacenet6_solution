#!/bin/bash

source activate solaris

TRAIN_DIR=$1  # path/to/spacenet6/train/AOI_11_Rotterdam/
source settings.sh

# preprocess dataset
./tools/compute_mean_std.py \
    --data_dir ${TRAIN_DIR} \
    --image_subdir SAR-Intensity \
    --out_dir ${IMAGE_MEAN_STD_DIR}

./tools/compute_mean_std.py \
    --data_dir ${TRAIN_DIR} \
    --image_subdir PS-RGBNIR \
    --out_dir ${IMAGE_MEAN_STD_DIR}

./tools/geojson_to_mask.py \
    --data_dir ${TRAIN_DIR} \
    --out_dir ${BUILDING_MASK_DIR} \
    --boundary_width ${BUILDING_BOUNDARY_WIDTH} \
    --min_area ${BUILDING_MIN_AREA}

./tools/split_dataset.py \
    --data_dir ${TRAIN_DIR} \
    --mask_dir ${BUILDING_MASK_DIR} \
    --out_dir ${DATA_SPLIT_DIR} \
    --split_num ${DATA_SPLIT_NUM}
