#!/bin/bash

ENV=desktop  # "desktop" or "mac"
if [ $# -eq 1 ]; then
    ENV=$1
fi

# set environment specific parameters
if [ $ENV = desktop ]; then
	RUNTIME="--runtime nvidia"
	FEATURE_ROOT=/mnt/sdb1/spacenet6/
elif [ $ENV = mac ]; then
	RUNTIME=""
	FEATURE_ROOT=${HOME}/features/spacenet6/
else
	echo 'Usage: ./run.sh $ENV'
	echo '(ENV must be "desktop" or "mac")'
	exit 1
fi

# set jupyter port
JUPYTER_PORT=8889
echo "mapping port docker:${JUPYTER_PORT} --> host:${JUPYTER_PORT}"

# set image name
IMAGE="spacenet6:dev"

# set project root dicrectory to map to docker
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# set path to directories to map to docker
DATA_DIR=${HOME}/data
WEIGHTS_DIR=${FEATURE_ROOT}/weights
LOG_DIR=${FEATURE_ROOT}/logs
PREDICTION_DIR=${FEATURE_ROOT}/predictions
ENSEMBLED_PREDICTION_DIR=${FEATURE_ROOT}/ensembled_predictions
POLY_CSV_DIR=${FEATURE_ROOT}/polygons
VAL_PREDICTION_DIR=${FEATURE_ROOT}/val_predictions
VAL_POLY_CSV_DIR=${FEATURE_ROOT}/val_polygons

# run container
CONTAINER="spacenet6_dev"

docker run ${RUNTIME} -it --rm --ipc=host \
	-p ${JUPYTER_PORT}:${JUPYTER_PORT} \
	-p 6006:6006 \
	-v ${PROJ_DIR}:/work \
	-v ${DATA_DIR}:/data \
	-v ${WEIGHTS_DIR}:/weights \
	-v ${LOG_DIR}:/logs \
	-v ${PREDICTION_DIR}:/predictions \
	-v ${ENSEMBLED_PREDICTION_DIR}:/ensembled_predictions \
	-v ${POLY_CSV_DIR}:/polygons \
	-v ${VAL_PREDICTION_DIR}:/val_predictions \
	-v ${VAL_POLY_CSV_DIR}:/val_polygons \
	--name ${CONTAINER} \
	${IMAGE} /bin/bash
