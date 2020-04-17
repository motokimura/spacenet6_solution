#!/bin/bash

FEATURE_DIR=/work/features

# path to SAR_orientations.txt
SAR_ORIENTATION_PATH=/work/static/SAR_orientations.txt

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
CONFIG_A_1=/work/configs/unet-scse_efficientnet-b7_ps-rgbnir_v_01.yml
CONFIG_A_2=/work/configs/unet-scse_efficientnet-b7_v_01.yml
CONFIG_B_1=/work/configs/unet-scse_efficientnet-b7_v_01.yml
CONFIG_C_1=/work/configs/unet-scse_efficientnet-b7_v_03.yml

TRAIN_LOG_DIR=${FEATURE_DIR}/logs
MODEL_WEIGHT_DIR=${FEATURE_DIR}/weights
SAVE_CHECKPOINTS='False'
DUMP_GIT_INFO='False'

TRAIN_STDOUT_DIR=/work/stdout/train

# test_spacenet6_model.py
MODEL_PREDICTION_DIR=${FEATURE_DIR}/predictions

TEST_STDOUT_DIR=/work/stdout/test

# ensemble_models.py
CLASSES='["building_footprint","building_boundary"]'
ENSEMBLED_PREDICTION_DIR=${FEATURE_DIR}/predictions_ensembled

# pred_array_to_poly.py
POLY_CSV_DIR=${FEATURE_DIR}/polygons
BOUNDARY_SUBSTRACT_COEFF=0.2
METHOD_TO_MAKE_POLYGONS='watershed'  # ['contours', 'watershed']
BUILDING_MIM_AREA_PIXEL=0  # for 'contours'
BUILDING_SCORE_THRESH=0.5  # for 'contours'
WATERSHED_MAIN_THRESH=0.3  # for 'watershed'
WATERSHED_SEED_THRESH=0.7  # for 'watershed'
WATERSHED_MIN_AREA_PIXEL=80  # for 'watershed'
WATERSHED_SEED_MIN_AREA_PIXEL=20  # for 'watershed'
