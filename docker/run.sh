#!/bin/bash

# set jupyter port
JUPYTER_PORT=8889
if [ $# -eq 1 ]; then
    JUPYTER_PORT=$1
fi

echo "mapping port docker:${JUPYTER_PORT} --> host:${JUPYTER_PORT}"

# set image name
IMAGE="spacenet6:dev"

# set project root dicrectory to map to docker
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# set path to directories to map to docker
DATA_DIR=${HOME}/data
WEIGHTS_DIR=/mnt/sdb1/spacenet6/weights
LOG_DIR=/mnt/sdb1/spacenet6/logs
PREDICTION_DIR=/mnt/sdb1/spacenet6/predictions
ENSEMBLED_PREDICTION_DIR=/mnt/sdb1/spacenet6/ensembled_predictions
POLY_CSV_DIR=/mnt/sdb1/spacenet6/polygons

# run container
CONTAINER="spacenet6_dev"

docker run --runtime=nvidia -it --rm --ipc=host \
	-p ${JUPYTER_PORT}:${JUPYTER_PORT} \
	-p 6006:6006 \
	-v ${PROJ_DIR}:/work \
	-v ${DATA_DIR}:/data \
	-v ${WEIGHTS_DIR}:/weights \
	-v ${LOG_DIR}:/logs \
	-v ${PREDICTION_DIR}:/predictions \
	-v ${ENSEMBLED_PREDICTION_DIR}:/ensembled_predictions \
	-v ${POLY_CSV_DIR}:/polygons \
	--name ${CONTAINER} \
	${IMAGE} /bin/bash
