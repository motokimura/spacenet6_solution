#!/bin/bash

# set jupyter port
JUPYTER_PORT=8889
if [ $# -eq 1 ]; then
    JUPYTER_PORT=$1
fi

echo "mapping port docker:${JUPYTER_PORT} --> host:${JUPYTER_PORT}"

# set image name
IMAGE="spacenet6:latest"

# set project root dicrectory to map to docker
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# set spacenet6 data directory to map to docker
DATA_DIR=${HOME}/data

# run container
CONTAINER="spacenet6"

docker run --runtime=nvidia -it --rm --ipc=host \
	-p ${JUPYTER_PORT}:${JUPYTER_PORT} \
	-v ${PROJ_DIR}:/work \
	-v ${DATA_DIR}:/data \
	--name ${CONTAINER} \
	${IMAGE} /bin/bash
