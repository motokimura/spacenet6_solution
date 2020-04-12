#!/bin/bash

source activate solaris

TRAIN_DIR=$1  # path/to/spacenet6/train/AOI_11_Rotterdam/
source settings.sh

# prepare directory to output intermediate products
mkdir -p ${FEATURE_DIR}

# prepare dataset
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

# train models
TRAIN_ARGS="\
    INPUT.TRAIN_VAL_SPLIT_DIR ${DATA_SPLIT_DIR} \
    INPUT.IMAGE_DIR ${TRAIN_DIR} \
    INPUT.BUILDING_DIR ${BUILDING_MASK_DIR} \
    INPUT.SAR_ORIENTATION ${SAR_ORIENTATION_PATH} \
    LOG_ROOT ${TRAIN_LOG_DIR} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
    CHECKPOINT_ROOT ${CHECKPOINT_DIR} \
"

## model-A (pretrain)
./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.IMAGE_TYPE PS-RGBNIR \
    SOLVER.EPOCHS 140 \  # TODO: inclide this in config
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 0

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.IMAGE_TYPE PS-RGBNIR \
    SOLVER.EPOCHS 140 \  # TODO: inclide this in config
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 1

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.IMAGE_TYPE PS-RGBNIR \
    SOLVER.EPOCHS 140 \  # TODO: inclide this in config
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 2

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.IMAGE_TYPE PS-RGBNIR \
    SOLVER.EPOCHS 140 \  # TODO: inclide this in config
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 3

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    INPUT.IMAGE_TYPE PS-RGBNIR \
    SOLVER.EPOCHS 140 \  # TODO: inclide this in config
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 4

## model-A (finetune)
./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0000/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 5

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0001/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 6

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0002/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 7

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0003/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 8

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_A} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0004/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 9

## model-A
./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_C} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 10

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_C} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 11

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_C} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 12

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_C} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 13

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_C} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 14

## model-B
./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_B} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 15

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_B} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 16

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_B} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 17

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_B} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 18

./tools/train_spacenet6_model.py \
    --config ${CONFIG_MODEL_B} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 19
