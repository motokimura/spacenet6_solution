#!/bin/bash

# set image name
IMAGE="spacenet6_data:latest"
if [ $# -eq 1 ]; then
    IMAGE=$1
fi

# set project root dicrectory to map to docker
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# set spacenet6 data directory to map to docker
DATA_DIR=${HOME}/data

# run container
CONTAINER="spacenet6_data"

docker run -it --rm --ipc=host \
	-p 8888:8888 -p 6006:6006 \
	-v ${PROJ_DIR}:/work \
	-v ${DATA_DIR}:/data \
	--name ${CONTAINER} \
	${IMAGE} /bin/bash
