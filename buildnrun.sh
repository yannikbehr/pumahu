#!/bin/bash

#########################################
# Build and run docker image            #
# 09/20 Y. Behr <y.behr@gns.cri.nz>     #
#########################################

IMAGE="huta17-d:5000/yannik/pumahu"
TAG=0.0.1
JUPYTER=false
BUILD=false
PUSH=false
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
    --image                 Provide alternative image name.
    --tag                   Provide alternative tag
    --push                  Push to registry. Note: the registry
                            has to be part of the image name
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
        --push) PUSH=true;;
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

if [ "${PUSH}" != "false" ]; then
    docker image push "${IMAGE}:${TAG}"
    docker image push "${IMAGE}_nginx:${TAG}"
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

