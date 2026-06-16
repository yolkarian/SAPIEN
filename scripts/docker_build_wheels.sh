#!/bin/bash

docker run -v `pwd`:/workspace/SAPIEN -it --rm \
       -u $(id -u ${USER}):$(id -g ${USER}) \
       -e SAPIEN_PHYSX5_DIR \
       -e SAPIEN_PHYSX5_GPU_DIR \
       -e SAPIEN_PHYSX5_VERSION \
       yolkarian/sapien-build-env:0.3 bash -c \
       "export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} && cd /workspace/SAPIEN && ./scripts/build.sh $1 --profile"
