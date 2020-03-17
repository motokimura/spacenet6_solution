#!/bin/bash

source activate solaris

TEST_DIR=$1  # path/to/spacenet6/test/
OUTPUT_CSV_PATH=$2  # path/to/solution.csv
TEST_IMAGE_DIR=${TEST_DIR}/SAR-Intensity

source settings.sh

# predict with trained models
TEST_ARGS="\
    INPUT.TEST_IMAGE_DIR ${TEST_IMAGE_DIR} \
    INPUT.SAR_ORIEENTATION ${SAR_ORIENTATION_PATH} \
    LOG_ROOT ${TRAIN_LOG_DIR} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
    PREDICTION_ROOT ${MODEL_PREDICTION_DIR} \
"

./tools/test_spacenet6_model.py \
    --exp_id 0 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 1 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 2 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 3 \
    ${TEST_ARGS}

./tools/test_spacenet6_model.py \
    --exp_id 4 \
    ${TEST_ARGS}

# ensemble
./tools/ensemble_models.py \
    INPUT.TEST_IMAGE_DIR ${TEST_IMAGE_DIR} \
    INPUT.CLASSES ${CLASSES} \
    PREDICTION_ROOT ${MODEL_PREDICTION_DIR} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_DIR} \
    ENSEMBLE_EXP_IDS "[0, 1, 2, 3, 4]"

# prediction to poly
./tools/pred_array_to_poly.py \
    INPUT.CLASSES ${CLASSES} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_DIR} \
    POLY_CSV_ROOT ${POLY_CSV_DIR} \
    BOUNDARY_SUBSTRACT_COEFF ${BOUNDARY_SUBSTRACT_COEFF} \
    BUILDING_MIM_AREA_PIXEL ${BUILDING_MIM_AREA_PIXEL} \
    BUILDING_SCORE_THRESH ${BUILDING_SCORE_THRESH} \
    ENSEMBLE_EXP_IDS "[0, 1, 2, 3, 4]"
