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

mkdir -p ${TEST_STDOUT_DIR}

## CONFIG_A_2
nohup CUDA_VISIBLE_DEVICES=0 ./tools/test_spacenet6_model.py \
    --exp_id 5 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0005.out &

nohup CUDA_VISIBLE_DEVICES=1 ./tools/test_spacenet6_model.py \
    --exp_id 6 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0006.out &

nohup CUDA_VISIBLE_DEVICES=2 ./tools/test_spacenet6_model.py \
    --exp_id 7 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0007.out &

nohup CUDA_VISIBLE_DEVICES=3 ./tools/test_spacenet6_model.py \
    --exp_id 8 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0008.out &

wait

nohup CUDA_VISIBLE_DEVICES=0 ./tools/test_spacenet6_model.py \
    --exp_id 9 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0009.out &

## CONFIG_B_1
nohup CUDA_VISIBLE_DEVICES=1 ./tools/test_spacenet6_model.py \
    --exp_id 10 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0010.out &

nohup CUDA_VISIBLE_DEVICES=2 ./tools/test_spacenet6_model.py \
    --exp_id 11 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0011.out &

nohup CUDA_VISIBLE_DEVICES=3 ./tools/test_spacenet6_model.py \
    --exp_id 12 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0012.out &

wait

nohup CUDA_VISIBLE_DEVICES=0 ./tools/test_spacenet6_model.py \
    --exp_id 13 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0013.out &

nohup CUDA_VISIBLE_DEVICES=1 ./tools/test_spacenet6_model.py \
    --exp_id 14 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0014.out &

## CONFIG_C_1
nohup CUDA_VISIBLE_DEVICES=2 ./tools/test_spacenet6_model.py \
    --exp_id 15 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0015.out &

nohup CUDA_VISIBLE_DEVICES=3 ./tools/test_spacenet6_model.py \
    --exp_id 16 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0016.out &

wait

nohup CUDA_VISIBLE_DEVICES=0 ./tools/test_spacenet6_model.py \
    --exp_id 17 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0017.out &

nohup CUDA_VISIBLE_DEVICES=1 ./tools/test_spacenet6_model.py \
    --exp_id 18 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0018.out &

nohup CUDA_VISIBLE_DEVICES=2 ./tools/test_spacenet6_model.py \
    --exp_id 19 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0019.out &

wait