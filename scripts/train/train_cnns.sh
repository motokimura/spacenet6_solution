#!/bin/bash

source activate solaris

TRAIN_DIR=$1  # path/to/spacenet6/train/AOI_11_Rotterdam/
source settings.sh

TRAIN_ARGS="\
    INPUT.TRAIN_VAL_SPLIT_DIR ${DATA_SPLIT_DIR} \
    INPUT.IMAGE_DIR ${TRAIN_DIR} \
    INPUT.BUILDING_DIR ${BUILDING_MASK_DIR} \
    INPUT.SAR_ORIENTATION ${SAR_ORIENTATION_PATH} \
    LOG_ROOT ${TRAIN_LOG_DIR} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
    SAVE_CHECKPOINTS ${SAVE_CHECKPOINTS} \
"
# comment out the line below for debug
#TRAIN_ARGS=${TRAIN_ARGS}" SOLVER.EPOCH 1"

## CONFIG_A_1
./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 0

./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 1

./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 2

./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 3

./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 4

## CONFIG_A_2 (finetune CONFIG_A_1 models)
./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0000/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 5

./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0001/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 6

./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0002/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 7

./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0003/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 8

./tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0004/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 9

## CONFIG_B_1
./tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 10

./tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 11

./tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 12

./tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 13

./tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 14

## CONFIG_C_1
./tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 15

./tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 16

./tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 17

./tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 18

./tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 19
