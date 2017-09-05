#!/bin/bash

##############################################
# Build docker image and start container     #
##############################################

docker rmi yadabe/clemb
docker build -t yadabe/clemb .
