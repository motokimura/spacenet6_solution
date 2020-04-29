#!/bin/bash

START_TIME=$SECONDS

source activate solaris

rm -rf ${MODEL_PREDICTION_DIR} ${ENSEMBLED_PREDICTION_DIR} ${POLY_CSV_DIR}

TEST_DIR=$1  # path/to/spacenet6/test/AOI_11_Rotterdam/
OUTPUT_CSV_PATH=$2  # path/to/solution.csv

source /work/settings.sh

# predict with trained models
/work/scripts/test/test_cnns.sh ${TEST_DIR}

# postprocess predictions
/work/scripts/test/postprocess.sh ${TEST_DIR} ${OUTPUT_CSV_PATH}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo 'Total time for testing: ' $(($ELAPSED_TIME/60+1)) '[min]'
