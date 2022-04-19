#!/bin/bash

set -e

###### Standard section to container images ####
PARAMS=""

while (( "$#" )); do
  case "$1" in
    -i|--image)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        IMAGE=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -h|--help)
      echo "Some text explaining the options"
      exit 0
      ;;
    -*) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"

## Find target server
if [ -z "$IMAGE" ]
then
  echo "Image name is required"
  exit 1
fi

echo "Building $IMAGE"

DOCKER_BUILDKIT=1 docker build . -t "$IMAGE"

echo "Pushing $IMAGE"
docker image push "$IMAGE"

exit 0
