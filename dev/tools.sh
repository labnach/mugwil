#!/bin/bash

# global vars.
PROJECT_NAME=$(basename `git config --get remote.origin.url` | sed 's/.git$//')
PROJECT_ROOT_DIR=`git rev-parse --show-toplevel`
PROJECT_VERSION=`cat ${PROJECT_ROOT_DIR}/VERSION`

cmd=$1
shift

case $cmd in
    b | build)
        docker build -t ${PROJECT_NAME}:${PROJECT_VERSION} ${PROJECT_ROOT_DIR}
        ;;
    r | run)
        docker run --rm -it --runtime=nvidia -p 8888:8888 \
            -v ${PROJECT_ROOT_DIR}/src:/src \
            ${PROJECT_NAME}:${PROJECT_VERSION} "$@"
        ;;
    h | help)
        cat ${PROJECT_ROOT_DIR}/dev/tools.sh
        ;;
    "" | *)
        echo "Bad command. Options are:"
        grep -E "^    . \| .*\)$" ${PROJECT_ROOT_DIR}/dev/tools.sh
    ;;
esac
