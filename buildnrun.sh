#!/bin/bash

#########################################
# Build and run docker image            #
# 09/20 Y. Behr <y.behr@gns.cri.nz>     #
#########################################

IMAGE=pumahu
TAG=0.1
BUILD=false
INTERACTIVE=false

function usage(){
cat <<EOF
Usage: $0 [Options] 
Build and run docker for ashfall visualisation.

Optional Arguments:
    -h, --help              Show this message.
    -b, --build             Rebuild the image.
    -i, --interactive       Start the container with a bash prompt.
    --image                 Provide alternative image name.
    --tag                   Provide alternative tag
EOF
}

# Processing command line options
while [ $# -gt 0 ]
do
    case "$1" in
        -b | --build) BUILD=true;;
        -i | --interactive) INTERACTIVE=true;;
        --image) IMAGE="$2";shift;;
        --tag) TAG="$2";shift;;
        -h) usage; exit 0;;
        -*) usage; exit 1;;
esac
shift
done


if [ "${BUILD}" == "true" ]; then
    docker rmi "${IMAGE}:${TAG}"
    docker build --build-arg NB_USER=$(whoami) \
        --build-arg NB_UID=$(id -u) \
        -t "${IMAGE}:${TAG}" .
fi

if [ "${INTERACTIVE}" == "true" ]; then
    docker run -it --rm -v $PWD:/home/$(whoami)/pumahu \
        -u $(id -u):$(id -g) \
        "${IMAGE}:${TAG}" /bin/bash
fi
