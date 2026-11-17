#!/bin/bash

rm -rf sapien.egg-info
rm -f dist/*.whl

python3 setup.py bdist_wheel

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd dist
pip3 uninstall -y sapien
# dist/ may also hold the C++ SDK archive of scripts/build.sh; install only wheels
pip3 install *.whl

cd /tmp && rm stubs -rf && python3 ${DIR}/python/stubgen.py sapien.core --ignore-invalid all
cp /tmp/stubs/sapien/core-stubs/__init__.pyi $DIR/python/py_package/core
cp -r /tmp/stubs/sapien/core-stubs/pysapien $DIR/python/py_package/core
