#!/bin/bash

######## Set up default project variables #######
IMAGE='huta17-d.gns.cri.nz:5000/yannik/pumahu:0.0.1'
APP_NAME=Pumahu
APP_SERVER="Vulkan"
TEAM="Volcano"
#################################################
####### Docker config ###########################
DOCKER_CONFIG=(HostConfig:='{"RestartPolicy": {"Name":"always" } , "Binds": [ "pumahu_data:/opt/data" ] }' \
Cmd:='["python", "./job_scheduler.py"]')
#################################################
####### Aliases #################################
PORTAINER_HOST='portainer:9000'
STD_HTTPS_OPTS="--verify no --ignore-stdin"
STATUS_HTTPS_OPTS="--check-status $STD_HTTPS_OPTS"
#################################################

set -e

###### Standard section to deploy containers ####
PARAMS=""

while (( "$#" )); do
  case "$1" in
    -s|--server)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        APP_SERVER=$2
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
    -d|--dockerConfig)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        DOCKER_CONFIG=("$2")
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
if [ -z "$APP_SERVER" ]
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
  echo "Building as $portainer_user and deploying to $APP_SERVER"
fi

TOKEN=$(https POST "$PORTAINER_HOST"/api/auth Username="$portainer_user" Password="$portainer_auth" $STD_HTTPS_OPTS | jq .jwt -r)

if [[ -z "$TOKEN" ]]
then
  echo "unauthorised access"
  exit 3
fi

AUTH="Authorization: Bearer $TOKEN"
## Find the target server id
APP_SERVER_ID=$( https GET "$PORTAINER_HOST"/api/endpoints "$AUTH" $STD_HTTPS_OPTS -b | jq --arg APP_SERVER "$APP_SERVER" '.[] | select(.Name == $APP_SERVER) | .Id')

if [[ -z "$APP_SERVER_ID" ]]
then
  echo "Server $APP_SERVER unavailable"
  exit 4
fi

## Find the owning team id
TEAM_ID=$(https GET "$PORTAINER_HOST"/api/teams "$AUTH" $STD_HTTPS_OPTS -b | jq --arg TEAM "$TEAM" '.[] | select(.Name == $TEAM) | .Id')
if [[ -z "$TEAM_ID" ]]
then
  echo "Team $TEAM unavailable"
  exit 4
fi

echo "Cleaning up old versions of $APP_NAME"
CONTAINERS=$(https GET "$PORTAINER_HOST"/api/endpoints/"$APP_SERVER_ID"/docker/containers/json?filters=\{\"name\":\{\""/$APP_NAME"\":true\}\} "$AUTH" $STD_HTTPS_OPTS -b)
echo "Found running containers $CONTAINERS"

if [[ ! -z $(echo "$CONTAINERS" | jq .[]? ) ]]
then
  IDS=$(echo "$CONTAINERS" | jq -r '.[] | .Id')
  for id in $IDS
  do
    echo "Stopping $id"
    if https POST "$PORTAINER_HOST"/api/endpoints/"$APP_SERVER_ID"/docker/containers/"$id"/stop "$AUTH" $STATUS_HTTPS_OPTS &> /dev/null
    then
      STOP_STATUS=$(https POST "$PORTAINER_HOST"/api/endpoints/"$APP_SERVER_ID"/docker/containers/"$id"/wait "$AUTH" $STATUS_HTTPS_OPTS -b)
      echo "Stopped $id with status $STOP_STATUS"
      echo "Removing $id"
      if https DELETE "$PORTAINER_HOST"/api/endpoints/"$APP_SERVER_ID"/docker/containers/"$id" "$AUTH" $STATUS_HTTPS_OPTS &> /dev/null
      then
        echo "Removed $id"
      else
          case $? in
              2) echo 'Request timed out!' ;;
              3) echo 'Unexpected HTTP 3xx Redirection!' ;;
              4) echo "Cannot remove $id" ;;
              5) echo 'HTTP 5xx Server Error!' ;;
              6) echo 'Exceeded --max-redirects=<n> redirects!' ;;
              *) echo 'Other Error!' ;;
          esac
      fi
    else
        case $? in
            2) echo 'Request timed out!' ;;
            3) echo 'Unexpected HTTP 3xx Redirection!' ;;
            4) echo 'HTTP 4xx Client Error!' ;;
            5) echo 'HTTP 5xx Server Error!' ;;
            6) echo 'Exceeded --max-redirects=<n> redirects!' ;;
            *) echo 'Other Error!' ;;
        esac
    fi
  done
else
  echo "No containers found $CONTAINERS"
fi

echo "Creating $IMAGE"
CREATE_RESP=$(https -f -b POST "$PORTAINER_HOST"/api/endpoints/"$APP_SERVER_ID"/docker/images/create "fromImage=$IMAGE" "$AUTH" $STD_HTTPS_OPTS)

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

CONTAINER_CREATE=$(https -b POST "$PORTAINER_HOST"/api/endpoints/"$APP_SERVER_ID"/docker/containers/create \
name=="$APP_NAME" \
Image="$IMAGE" \
"${DOCKER_CONFIG[@]}" \
"$AUTH" --verify no --ignore-stdin )

if [[ -z $(echo "$CONTAINER_CREATE" | jq .Id ) ]]
then
  echo "Could not create container $(jq -r .message)"
  echo "Could not create container $(jq -r .Warnings)"
  exit 6
else
  NEW_APP=$(echo "$CONTAINER_CREATE" | jq -r .Id)
  echo "Created $APP_NAME with ID: $NEW_APP"

  if https POST "$PORTAINER_HOST"/api/endpoints/"$APP_SERVER_ID"/docker/containers/"$NEW_APP"/start "$AUTH" --verify no --ignore-stdin --check-status &> /dev/null
  then
    echo "Started $APP_NAME on $APP_SERVER"
  else
      case $? in
          2) echo 'Request timed out!' ;;
          3) echo 'Unexpected HTTP 3xx Redirection!' ;;
          4) echo "Cannot remove $id" ;;
          5) echo 'HTTP 5xx Server Error!' ;;
          6) echo 'Exceeded --max-redirects=<n> redirects!' ;;
          *) echo 'Other Error!' ;;
      esac
  fi

  RET=$?
  if [[ $RET -eq 0 ]]
  then
    ## Ensure the correct team can access the application in Portainer by updating the ownership
    RESOURCE_ID=$(echo "$CONTAINER_CREATE" | jq -r '.Portainer.ResourceControl.Id' )

    if https PUT "$PORTAINER_HOST"/api/resource_controls/"$RESOURCE_ID" Teams:=["$TEAM_ID"] "$AUTH" --verify no --ignore-stdin -b &> /dev/null
    then
      echo "Assigned to $TEAM"
    else
        case $? in
            2) echo 'Request timed out!' ;;
            3) echo 'Unexpected HTTP 3xx Redirection!' ;;
            4) echo "Cannot assign to $TEAM" ;;
            5) echo 'HTTP 5xx Server Error!' ;;
            6) echo 'Exceeded --max-redirects=<n> redirects!' ;;
            *) echo 'Other Error!' ;;
        esac
    fi
  else
      case $RET in
          2) echo 'Request timed out!' ;;
          3) echo "$APP_NAME already started";;
          4) echo "No such container $id" ;;
          5) echo 'HTTP 5xx Server Error!' ;;
          6) echo 'Exceeded --max-redirects=<n> redirects!' ;;
          *) echo 'Other Error!' ;;
      esac
      exit 8
  fi
fi

exit 0
