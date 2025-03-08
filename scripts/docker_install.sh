#!/bin/bash

[[ `python -V` =~ ^Python\ 3\.([0-9]+)\..*$ ]] || echo failed to detect Python version

PYTHON_VERSION=3${BASH_REMATCH[1]}

echo Python ${PYTHON_VERSION} detected

rm -rf sapien.egg-info
rm -f wheelhouse/*.whl

docker run -v `pwd`:/workspace/SAPIEN -it --rm \
       -u $(id -u ${USER}):$(id -g ${USER}) \
       yolkarian/sapien-build-env:0.2 bash -c "export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} && cd /workspace/SAPIEN && ./scripts/build.sh ${PYTHON_VERSION}"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd wheelhouse
pip3 uninstall -y sapien
pip3 install *

cd /tmp && rm stubs -rf && python3 ${DIR}/python/stubgen.py sapien
cp -r /tmp/stubs/sapien/__init__.pyi ${DIR}/python/py_package
cp -r /tmp/stubs/sapien/pysapien ${DIR}/python/py_package
