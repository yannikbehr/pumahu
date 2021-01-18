#!/bin/bash

#########################################
# Build and run docker image            #
# 09/20 Y. Behr <y.behr@gns.cri.nz>     #
#########################################
##############################################
IMAGE=pumahu
##############################################

IMAGENAME=pumahu
TAG=latest
JUPYTER=false
BUILD=false
NOCACHE=false
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
        -h) usage; exit 0;;
        -*) usage; exit 1;;
esac
shift
done


if [ "${BUILD}" == "true" ]; then
    docker build --build-arg NB_USER=$(whoami) \
        --build-arg NB_UID=$(id -u) \
        -t "${IMAGENAME}:${TAG}" .
fi

if [ "${INTERACTIVE}" == "true" ]; then
    docker run -it --rm -v $PWD:/home/$(whoami)/pumahu \
        -u $(id -u):$(id -g) \
        "$IMAGENAME:$TAG" /bin/bash
fi

   
if [ "$JUPYTER" == "true" ];then
    docker run -it --rm -v $PWD:/home/$(whoami)/pumahu \
        -u $(id -u):$(id -g) \
        -p $JPORT:$JPORT \
        -w /home/$(whoami) \
        "$IMAGENAME:$TAG" \
        jupyter-lab --ip 0.0.0.0 --no-browser --port $JPORT
fi

