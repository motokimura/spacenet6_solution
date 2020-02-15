#!/bin/bash

# Set image name
IMAGE="spacenet6_data:latest"

# Get project root dicrectory
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# Buld docker image from project root directory
cd ${PROJ_DIR} && \
docker build -t ${IMAGE} -f ${THIS_DIR}/Dockerfile .
