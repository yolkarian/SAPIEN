#!/bin/bash

docker run -v `pwd`:/workspace/SAPIEN -it --rm \
       -u $(id -u ${USER}):$(id -g ${USER}) \
       yolkarian/sapien-build-env:0.2 bash -c \
       "export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} && cd /workspace/SAPIEN && ./scripts/build.sh $1 --profile"
