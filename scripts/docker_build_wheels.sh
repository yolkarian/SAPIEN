#!/bin/bash
# Builds SAPIEN in the build image: the wheels into wheelhouse/ and the C++ SDK into
# dist/sapien-sdk-<version>-linux_x86_64.tar.gz, which scripts/build.sh tests on CPU PhysX before it
# packs the wheels. Arguments go to scripts/build.sh, e.g. the Python version (311).
#
# For local builds, CMAKE_BUILD_PARALLEL_LEVEL=4 is recommended. This wrapper
# only forwards the caller's setting and intentionally does not impose a cap.
DOCKER_ENV_ARGS=()
if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
       DOCKER_ENV_ARGS+=("-e" "CMAKE_BUILD_PARALLEL_LEVEL")
fi
# a terminal only when there is one, so the build also runs from ssh, tmux scripts or CI
DOCKER_TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
       DOCKER_TTY_ARGS+=("-it")
fi

docker run -v `pwd`:/workspace/SAPIEN "${DOCKER_TTY_ARGS[@]}" --rm \
       -u $(id -u ${USER}):$(id -g ${USER}) \
       "${DOCKER_ENV_ARGS[@]}" \
       -e SAPIEN_PHYSX5_DIR \
       -e SAPIEN_PHYSX5_GPU_DIR \
       -e SAPIEN_PHYSX5_VERSION \
       yolkarian/sapien-build-env:0.3.1 bash -lc \
       'cd /workspace/SAPIEN && ./scripts/build.sh "$@" --profile' bash "$@" || exit $?

ls -l wheelhouse/*.whl dist/sapien-sdk-*.tar.gz
echo "Check the SDK on GPU PhysX on a machine with an NVIDIA GPU:"
echo "  ./scripts/test_sdk.sh dist/sapien-sdk-*-linux_x86_64.tar.gz --gpu"
