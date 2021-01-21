#!/bin/bash

set -e

######## Set up default project variables #######
IMAGE='huta17-d.gns.cri.nz:5000/yannik/pumahu_nginx:0.0.1'
APP_NAME=Pumahu-nginx
PORTAINER_HOST='huta17-d:9000'
SERVER="Vulkan"
TEAM="Volcano"
#################################################
####### Docker config ###########################
DOCKER_CONFIG=(HostConfig:='{ "PortBindings": { "80/tcp": [{ "HostPort": "9080" }] }, "RestartPolicy": {"Name":"always" } \
    "Binds": [ "pumahu_data:/usr/share/nginx/html:ro" ] }' \
ExposedPorts:='{ "80/tcp": {} }' )
#################################################

###### Standard section to deploy containers ####
PARAMS=""

while (( "$#" )); do
  case "$1" in
    -s|--server)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        SERVER=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -t|--team)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        TEAM=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -i|--image)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        IMAGE=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -a|--appname)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        APP_NAME=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
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
if [ -z "$SERVER" ]
then
  echo "Server is undefined"
  exit 1
fi

# Get the API auth token
if [[ -z "$portainer_user" || -z "$portainer_auth" ]]
then
  echo "Environment variables undefined"
  exit 2
else
  echo "Building as $portainer_user and deploying to $SERVER"
fi

TOKEN=$(http POST "$PORTAINER_HOST"/api/auth Username="$portainer_user" Password="$portainer_auth" --ignore-stdin  | jq .jwt -r)

if [[ -z "$TOKEN" ]]
then
  echo "unauthorised access"
  exit 3
fi
## Find the target server id
SERVER_ID=$( http GET "$PORTAINER_HOST"/api/endpoints "Authorization: Bearer $TOKEN" --ignore-stdin  -b | jq --arg SERVER "$SERVER" '.[] | select(.Name == $SERVER) | .Id')

if [[ -z "$SERVER_ID" ]]
then
  echo "Server $SERVER unavailable"
  exit 4
fi

## Find the owning team id
TEAM_ID=$(http GET "$PORTAINER_HOST"/api/teams "Authorization: Bearer $TOKEN" --ignore-stdin  -b | jq --arg TEAM "$TEAM" '.[] | select(.Name == $TEAM) | .Id')
if [[ -z "$TEAM_ID" ]]
then
  echo "Team $TEAM unavailable"
  exit 4
fi

echo "Cleaning up old versions of $APP_NAME"
CONTAINERS=$(http GET "$PORTAINER_HOST"/api/endpoints/"$SERVER_ID"/docker/containers/json?filters=\{\"name\":\{\""/$APP_NAME"\":true\}\} "Authorization: Bearer $TOKEN" --ignore-stdin  -b)
echo "Found running containers $CONTAINERS"

if [[ ! -z $(echo "$CONTAINERS" | jq .[]? ) ]]
then
  IDS=$(echo "$CONTAINERS" | jq -r '.[] | .Id')
  for id in $IDS
  do
    echo "Stopping $id"
    if http POST "$PORTAINER_HOST"/api/endpoints/"$SERVER_ID"/docker/containers/"$id"/stop "Authorization: Bearer $TOKEN" --check-status --ignore-stdin &> /dev/null
    then
      STOP_STATUS=$(http POST "$PORTAINER_HOST"/api/endpoints/"$SERVER_ID"/docker/containers/"$id"/wait "Authorization: Bearer $TOKEN" --check-status --ignore-stdin -b)
      echo "Stopped $id with status $STOP_STATUS"
      echo "Removing $id"
      if http DELETE "$PORTAINER_HOST"/api/endpoints/"$SERVER_ID"/docker/containers/"$id" "Authorization: Bearer $TOKEN" --check-status --ignore-stdin &> /dev/null
      then
        echo "Removed $id"
      else
          case $? in
              4) echo "Cannot remove $id" ;;
              5) echo 'HTTP 5xx Server Error!' ;;
              *) echo 'Other Error!' ;;
          esac
      fi
    else
        case $? in
            3) echo "$id already stopped" ;;
            4) echo "$id does not exist" ;;
            5) echo 'HTTP 5xx Server Error!' ;;
            *) echo 'Other Error!' ;;
        esac
    fi
  done
else
  echo "No containers found $CONTAINERS"
fi

echo "Creating $IMAGE"
CREATE_RESP=$(http -f POST "$PORTAINER_HOST"/api/endpoints/"$SERVER_ID"/docker/images/create "Authorization: Bearer $TOKEN" "fromImage=$IMAGE" --ignore-stdin -b)

if [[ -z  "$CREATE_RESP" ]]
then
  echo "No response when trying to create the image for $IMAGE"
  exit 5
else
  IFS=$'\n'
  for st in $CREATE_RESP
  do
    echo "$st" | jq .status -r
  done
fi

echo "Starting container $APP_NAME"

CONTAINER_CREATE=$(http POST "$PORTAINER_HOST"/api/endpoints/"$SERVER_ID"/docker/containers/create "Authorization: Bearer $TOKEN" \
name=="$APP_NAME" \
Image="$IMAGE" \
"${DOCKER_CONFIG[@]}" \
--ignore-stdin -b)

if [[ -z $(echo "$CONTAINER_CREATE" | jq .Id ) ]]
then
  echo "Could not create container $(jq -r .message)"
  echo "Could not create container $(jq -r .Warnings)"
  exit 6
else
  NEW_APP=$(echo "$CONTAINER_CREATE" | jq -r .Id)
  http POST "$PORTAINER_HOST"/api/endpoints/"$SERVER_ID"/docker/containers/"$NEW_APP"/start "Authorization: Bearer $TOKEN" --check-status --ignore-stdin &> /dev/null
  RET=$?
  if [[ $RET -eq 0 ]]
  then
    echo "Starting $APP_NAME - $id on $SERVER"
    ## Ensure the correct team can access the application in Portainer by updating the ownership
    RESOURCE_ID=$(echo "$CONTAINER_CREATE" | jq -r '.Portainer.ResourceControl.Id' )
    http PUT "$PORTAINER_HOST"/api/resource_controls/"$RESOURCE_ID" "Authorization: Bearer $TOKEN" Teams:=["$TEAM_ID"] --ignore-stdin -b &> /dev/null
  else
      case $RET in
        3) echo "$APP_NAME already started";;
        4) echo "No such container $id" ;;
        5) echo 'HTTP 5xx Server Error!' ;;
        *) echo 'Other Error!' ;;
      esac
      exit 8
  fi
fi

exit 0
