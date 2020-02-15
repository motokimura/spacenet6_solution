#!/bin/bash
# launch jupyter from spacenet6_data container

# get project root dicrectory
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# build docker image from project root directory
cd ${PROJ_DIR} && \
jupyter lab --port 8888 --ip=0.0.0.0 --allow-root --no-browser
