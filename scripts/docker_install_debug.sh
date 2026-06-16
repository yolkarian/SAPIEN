#!/bin/bash

DOCKER_ENV_ARGS=()
if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
       DOCKER_ENV_ARGS+=("-e" "CMAKE_BUILD_PARALLEL_LEVEL")
fi

[[ `python -V` =~ ^Python\ 3\.([0-9]+)\..*$ ]] || echo failed to detect Python version

PYTHON_VERSION=3${BASH_REMATCH[1]}

echo Python ${PYTHON_VERSION} detected

BUILD_ARGS=("${PYTHON_VERSION}" "$@")

docker run -v `pwd`:/workspace/SAPIEN -it --rm \
       -u $(id -u ${USER}):$(id -g ${USER}) \
       "${DOCKER_ENV_ARGS[@]}" \
       -e SAPIEN_PHYSX5_DIR \
       -e SAPIEN_PHYSX5_GPU_DIR \
       -e SAPIEN_PHYSX5_VERSION \
       yolkarian/sapien-build-env:0.3.1 bash -lc \
       'cd /workspace/SAPIEN && ./scripts/build.sh "$@" --debug' bash "${BUILD_ARGS[@]}"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd wheelhouse
pip3 uninstall -y sapien
pip3 install *

cd /tmp && rm stubs -rf && python3 ${DIR}/python/stubgen.py sapien.core --ignore-invalid all
cp /tmp/stubs/sapien/core-stubs/__init__.pyi $DIR/python/py_package/core
cp -r /tmp/stubs/sapien/core-stubs/pysapien $DIR/python/py_package/core
