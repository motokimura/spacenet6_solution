#!/bin/bash

source activate solaris

TEST_DIR=$1  # path/to/spacenet6/test/AOI_11_Rotterdam/
TEST_IMAGE_DIR=${TEST_DIR}/SAR-Intensity

source settings.sh

# predict with trained models
TEST_ARGS="\
    INPUT.TEST_IMAGE_DIR ${TEST_IMAGE_DIR} \
    INPUT.SAR_ORIENTATION ${SAR_ORIENTATION_PATH} \
    LOG_ROOT ${TRAIN_LOG_DIR} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
    PREDICTION_ROOT ${MODEL_PREDICTION_DIR} \
"

## CONFIG_A_2
./tools/test_spacenet6_model.py \
    --exp_id 5 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 6 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 7 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 8 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 9 \
    ${TEST_ARGS}

## CONFIG_B_1
./tools/test_spacenet6_model.py \
    --exp_id 10 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 11 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 12 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 13 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 14 \
    ${TEST_ARGS}

## CONFIG_C_1
./tools/test_spacenet6_model.py \
    --exp_id 15 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 16 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 17 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 18 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 19 \
    ${TEST_ARGS}
