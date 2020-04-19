#!/bin/bash

source activate solaris

TEST_DIR=$1  # path/to/spacenet6/test/AOI_11_Rotterdam/
OUTPUT_CSV_PATH=$2  # path/to/solution.csv
TEST_IMAGE_DIR=${TEST_DIR}/SAR-Intensity

source settings.sh

# ensemble
./tools/ensemble_models.py \
    INPUT.TEST_IMAGE_DIR ${TEST_IMAGE_DIR} \
    INPUT.CLASSES ${CLASSES} \
    PREDICTION_ROOT ${MODEL_PREDICTION_DIR} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_DIR} \
    ENSEMBLE_EXP_IDS "[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]"

# prediction to poly
./tools/pred_array_to_poly.py \
    INPUT.CLASSES ${CLASSES} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_DIR} \
    POLY_CSV_ROOT ${POLY_CSV_DIR} \
    POLY_OUTPUT_PATH ${OUTPUT_CSV_PATH} \
    BOUNDARY_SUBSTRACT_COEFF ${BOUNDARY_SUBSTRACT_COEFF} \
    METHOD_TO_MAKE_POLYGONS ${METHOD_TO_MAKE_POLYGONS} \
    BUILDING_MIM_AREA_PIXEL ${BUILDING_MIM_AREA_PIXEL} \
    BUILDING_SCORE_THRESH ${BUILDING_SCORE_THRESH} \
    WATERSHED_MAIN_THRESH ${WATERSHED_MAIN_THRESH} \
    WATERSHED_SEED_THRESH ${WATERSHED_SEED_THRESH} \
    WATERSHED_MIN_AREA_PIXEL ${WATERSHED_MIN_AREA_PIXEL} \
    WATERSHED_SEED_MIN_AREA_PIXEL ${WATERSHED_SEED_MIN_AREA_PIXEL} \
    ENSEMBLE_EXP_IDS "[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]"
