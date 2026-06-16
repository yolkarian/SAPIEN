#!/bin/bash

DOCKER_ENV_ARGS=()
if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
       DOCKER_ENV_ARGS+=("-e" "CMAKE_BUILD_PARALLEL_LEVEL")
fi

docker run -v `pwd`:/workspace/SAPIEN -it --rm \
       -u $(id -u ${USER}):$(id -g ${USER}) \
       "${DOCKER_ENV_ARGS[@]}" \
       -e SAPIEN_PHYSX5_DIR \
       -e SAPIEN_PHYSX5_GPU_DIR \
       -e SAPIEN_PHYSX5_VERSION \
       yolkarian/sapien-build-env:0.3.1 bash -lc \
       'cd /workspace/SAPIEN && ./scripts/build.sh "$@" --profile' bash "$@"
