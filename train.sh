#!/bin/bash

source activate solaris

TRAIN_DIR=$1  # path/to/spacenet6/train/
source settings.sh

# prepare directory to output intermediate products
mkdir -p ${FEATUREE_DIR}

# copy SAR_orientations.txt for test.sh
cp ${TRAIN_DIR}/SummaryData/SAR_orientations.txt ${SAR_ORIENTATION_PATH}

# prepare dataset
./tools/compute_mean_std.py \
    --data_dir ${TRAIN_DIR} \
    --image_subdir SAR-Intensity \
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

# train models
TRAIN_ARGS="\
    INPUT.TRAIN_VAL_SPLIT_DIR ${DATA_SPLIT_DIR} \
    INPUT.IMAGE_DIR ${TRAIN_DIR} \
    INPUT.BUILDING_DIR ${BUILDING_MASK_DIR} \
    INPUT.SAR_ORIEENTATION ${SAR_ORIENTATION_PATH} \
    LOG_ROOT ${TRAIN_LOG_DIR} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
"

## model-A
./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 0

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 1

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 2

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 3

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 4
