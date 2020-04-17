#!/bin/bash

source activate solaris

TEST_DIR=$1  # path/to/spacenet6/test/AOI_11_Rotterdam/
TEST_IMAGE_DIR=${TEST_DIR}/SAR-Intensity

source settings.sh

# predict with trained models
TEST_ARGS="\
    --exp_log_dir ${TRAIN_LOG_DIR} \
    INPUT.TEST_IMAGE_DIR ${TEST_IMAGE_DIR} \
    INPUT.SAR_ORIENTATION ${SAR_ORIENTATION_PATH} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
    PREDICTION_ROOT ${MODEL_PREDICTION_DIR} \
"

mkdir -p ${TEST_STDOUT_DIR}

echo 'testing... (1/4)'

## CONFIG_A_2
nohup env CUDA_VISIBLE_DEVICES=0 ./tools/test_spacenet6_model.py \
    --exp_id 5 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0005.out &

nohup env CUDA_VISIBLE_DEVICES=1 ./tools/test_spacenet6_model.py \
    --exp_id 6 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0006.out &

nohup env CUDA_VISIBLE_DEVICES=2 ./tools/test_spacenet6_model.py \
    --exp_id 7 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0007.out &

nohup env CUDA_VISIBLE_DEVICES=3 ./tools/test_spacenet6_model.py \
    --exp_id 8 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0008.out &

wait

echo 'testing... (2/4)'

nohup env CUDA_VISIBLE_DEVICES=0 ./tools/test_spacenet6_model.py \
    --exp_id 9 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0009.out &

## CONFIG_B_1
nohup env CUDA_VISIBLE_DEVICES=1 ./tools/test_spacenet6_model.py \
    --exp_id 10 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0010.out &

nohup env CUDA_VISIBLE_DEVICES=2 ./tools/test_spacenet6_model.py \
    --exp_id 11 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0011.out &

nohup env CUDA_VISIBLE_DEVICES=3 ./tools/test_spacenet6_model.py \
    --exp_id 12 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0012.out &

wait

echo 'testing... (3/4)'

nohup env CUDA_VISIBLE_DEVICES=0 ./tools/test_spacenet6_model.py \
    --exp_id 13 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0013.out &

nohup env CUDA_VISIBLE_DEVICES=1 ./tools/test_spacenet6_model.py \
    --exp_id 14 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0014.out &

## CONFIG_C_1
nohup env CUDA_VISIBLE_DEVICES=2 ./tools/test_spacenet6_model.py \
    --exp_id 15 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0015.out &

nohup env CUDA_VISIBLE_DEVICES=3 ./tools/test_spacenet6_model.py \
    --exp_id 16 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0016.out &

wait

echo 'testing... (4/4)'

nohup env CUDA_VISIBLE_DEVICES=0 ./tools/test_spacenet6_model.py \
    --exp_id 17 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0017.out &

nohup env CUDA_VISIBLE_DEVICES=1 ./tools/test_spacenet6_model.py \
    --exp_id 18 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0018.out &

nohup env CUDA_VISIBLE_DEVICES=2 ./tools/test_spacenet6_model.py \
    --exp_id 19 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_DIR}/exp_0019.out &

wait

echo 'done tesging all models!'
