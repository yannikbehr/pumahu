#!/bin/bash

#########################################
# Build and run docker image            #
# 09/20 Y. Behr <y.behr@gns.cri.nz>     #
#########################################

IMAGE=pumahu
TAG=latest
JUPYTER=false
BUILD=false
REGISTRY=false
JPORT=8892

function usage(){
cat <<EOF
Usage: $0 [Options] 
Build and run docker for ashfall visualisation.

Optional Arguments:
    -h, --help              Show this message.
    -b, --build             Rebuild the image.
    -i, --interactive       Start the container with a bash prompt.
    -j, --jupyter           Start jupyter lab.
    
EOF
}

# Processing command line options
while [ $# -gt 0 ]
do
    case "$1" in
        -b | --build) BUILD=true;;
        -i | --interactive) INTERACTIVE=true;;
        -j | --jupyter) JUPYTER=true;;
        --image) IMAGE="$2";shift;;
        --tag) TAG="$2";shift;;
        --registry) REGISTRY="$2";shift;;
        -h) usage; exit 0;;
        -*) usage; exit 1;;
esac
shift
done


if [ "${BUILD}" == "true" ]; then
    docker build --build-arg NB_USER=$(whoami) \
        --build-arg NB_UID=$(id -u) \
        -t "${IMAGE}:${TAG}" .
    docker build -t "${IMAGE}_nginx:${TAG}" -f Dockerfile.nginx .
fi

if [ "${REGISTRY}" != "false" ]; then
    echo "push ${IMAGE} to ${REGISTRY}/yannik/${IMAGE}:${TAG}"
    docker image push "${REGISTRY}/yannik/${IMAGE}:${TAG}"
    echo "push ${IMAGE}_nginx to ${REGISTRY}/yannik/${IMAGE}_nginx:${TAG}"
    docker image push "${REGISTRY}/yannik/${IMAGE}_nginx:${TAG}"
fi

if [ "${INTERACTIVE}" == "true" ]; then
    docker run -it --rm -v $PWD:/home/$(whoami)/pumahu \
        -u $(id -u):$(id -g) \
        "${IMAGE}" /bin/bash
fi

   
if [ "$JUPYTER" == "true" ];then
    docker run -it --rm -v $PWD:/home/$(whoami)/pumahu \
        -u $(id -u):$(id -g) \
        -p $JPORT:$JPORT \
        -w /home/$(whoami) \
        "${IMAGE}" \
        jupyter-lab --ip 0.0.0.0 --no-browser --port $JPORT
fi

