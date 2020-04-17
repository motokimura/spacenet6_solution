#!/bin/bash

source activate solaris

TEST_DIR=$1  # path/to/spacenet6/test/AOI_11_Rotterdam/
OUTPUT_CSV_PATH=$2  # path/to/solution.csv
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

# ensemble
./tools/ensemble_models.py \
    INPUT.TEST_IMAGE_DIR ${TEST_IMAGE_DIR} \
    INPUT.CLASSES ${CLASSES} \
    PREDICTION_ROOT ${MODEL_PREDICTION_DIR} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_DIR} \
    ENSEMBLE_EXP_IDS "[5, 6, 7, 8, 9, 15, 16, 17, 18, 19]"

# prediction to poly
./tools/pred_array_to_poly.py \
    INPUT.CLASSES ${CLASSES} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_DIR} \
    POLY_CSV_ROOT ${POLY_CSV_DIR} \
    BOUNDARY_SUBSTRACT_COEFF ${BOUNDARY_SUBSTRACT_COEFF} \
    METHOD_TO_MAKE_POLYGONS ${METHOD_TO_MAKE_POLYGONS} \
    BUILDING_MIM_AREA_PIXEL ${BUILDING_MIM_AREA_PIXEL} \
    BUILDING_SCORE_THRESH ${BUILDING_SCORE_THRESH} \
    WATERSHED_MAIN_THRESH ${WATERSHED_MAIN_THRESH} \
    WATERSHED_SEED_THRESH ${WATERSHED_SEED_THRESH} \
    WATERSHED_MIN_AREA_PIXEL ${WATERSHED_MIN_AREA_PIXEL} \
    WATERSHED_SEED_MIN_AREA_PIXEL ${WATERSHED_SEED_MIN_AREA_PIXEL} \
    ENSEMBLE_EXP_IDS "[5, 6, 7, 8, 9, 15, 16, 17, 18, 19]"
