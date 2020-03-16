#!/bin/bash

FEATURE_DIR=/work/features

# copy SAR_orientations.txt
SAR_ORIENTATION_PATH=${FEATURE_DIR}/SAR_orientations.txt

# compute_mean_std.py
IMAGE_MEAN_STD_DIR=${FEATURE_DIR}/image_mean_std

# geojson_to_mask.py
BUILDING_MASK_DIR=${FEATURE_DIR}/masks
BUILDING_BOUNDARY_WIDTH=6
BUILDING_MIN_AREA=20

# split_dataset.py
DATA_SPLIT_DIR=${FEATURE_DIR}/split
DATA_SPLIT_NUM=5

# train_spacenet6_model.py
TRAIN_CONFIG=/work/configs/efficientnet-b5_v_01.yml
TRAIN_LOG_DIR=${FEATURE_DIR}/logs
MODEL_WEIGHT_DIR=${FEATURE_DIR}/weights

# test_spacenet6_model.py
MODEL_PREDICTION_DIR=${FEATURE_DIR}/predictions
