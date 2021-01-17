#!/bin/bash

#########################################
# Build and run docker image            #
# 01/21 Y. Behr <y.behr@gns.cri.nz>     #
#########################################

SERVICE=pumahu
JUPYTER=false
BUILD=false
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
    docker-compose build --build-arg NB_USER=$(whoami) \
        --build-arg NB_UID=$(id -u) 
fi

if [ "${INTERACTIVE}" == "true" ]; then
    docker-compose run --rm -v $PWD:/home/$(whoami)/pumahu \
        -u $(id -u):$(id -g) $SERVICE /bin/bash
fi

   
if [ "$JUPYTER" == "true" ];then
    docker-compose run --rm -v $PWD:/home/$(whoami)/pumahu \
        -u $(id -u):$(id -g) \
        -p $JPORT:$JPORT \
        -w /home/$(whoami) \
        ${SERVICE} \
        jupyter-lab --ip 0.0.0.0 --no-browser --port $JPORT
fi

