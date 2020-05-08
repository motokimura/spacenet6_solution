#!/bin/bash

source activate solaris

TEST_DIR=$1  # path/to/spacenet6/test/AOI_11_Rotterdam/
TEST_IMAGE_DIR=${TEST_DIR}/SAR-Intensity

source /work/settings.sh

# predict with trained models
TEST_ARGS="\
    --exp_log_dir ${TRAIN_LOG_DIR} \
    INPUT.TEST_IMAGE_DIR ${TEST_IMAGE_DIR} \
    INPUT.SAR_ORIENTATION ${SAR_ORIENTATION_PATH} \
    INPUT.MEAN_STD_DIR ${IMAGE_MEAN_STD_DIR} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
    PREDICTION_ROOT ${MODEL_PREDICTION_DIR} \
    DATALOADER.TEST_BATCH_SIZE 1 \
    DATALOADER.TEST_NUM_WORKERS 1 \
"

mkdir -p ${TEST_STDOUT_DIR}

echo ''
echo 'predicting... (1/4)'

## CONFIG_A_2
CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 5 \
    ${TEST_ARGS}

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 6 \
    ${TEST_ARGS}

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 7 \
    ${TEST_ARGS}

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 8 \
    ${TEST_ARGS}

echo 'predicting... (2/4)'

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 9 \
    ${TEST_ARGS}

## CONFIG_B_1
CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 10 \
    ${TEST_ARGS}

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 11 \
    ${TEST_ARGS}

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 12 \
    ${TEST_ARGS}

echo 'predicting... (3/4)'

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 13 \
    ${TEST_ARGS}

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 14 \
    ${TEST_ARGS}

## CONFIG_C_1
CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 15 \
    ${TEST_ARGS}

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 16 \
    ${TEST_ARGS}

echo 'predicting... (4/4)'

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 17 \
    ${TEST_ARGS}

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 18 \
    ${TEST_ARGS}

CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 19 \
    ${TEST_ARGS}

echo 'done predicting all models!'
