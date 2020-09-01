#!/bin/bash

if [[ "$MODE" = "cpu" ]]; then
    docker_run='docker run'
    maybe_gpu=''
else
    docker_run='docker run --runtime=nvidia'
    maybe_gpu='-gpu'
fi

project_name=$(basename `git config --get remote.origin.url` | sed 's/.git$//')
project_root_dir=`git rev-parse --show-toplevel`
project_version=`cat ${project_root_dir}/VERSION`

cmd=$1
shift

case ${cmd} in
    b | build)
        docker build --build-arg MAYBE_GPU=${maybe_gpu} \
            -t ${project_name}:${project_version}${maybe_gpu} \
            ${project_root_dir}
        ;;
    r | run)
        ${docker_run} --rm -it -p 8888:8888 \
            -v ${project_root_dir}/src:/src \
            ${project_name}:${project_version}${maybe_gpu} "$@"
        ;;
    h | help)
        cat ${project_root_dir}/dev/tools.sh
        ;;
    "" | *)
        echo "Bad command. Options are:"
        grep -E "^    . \| .*\)$" ${project_root_dir}/dev/tools.sh
    ;;
esac
