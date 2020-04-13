#!/bin/bash

source activate solaris

TEST_DIR=$1  # path/to/spacenet6/test/AOI_11_Rotterdam/
OUTPUT_CSV_PATH=$2  # path/to/solution.csv

source settings.sh

# predict with trained models
./scripts/test/test_cnns.sh ${TEST_DIR}

# postprocess predictions
./scripts/test/postprocess.sh ${TEST_DIR} ${OUTPUT_CSV_PATH}
