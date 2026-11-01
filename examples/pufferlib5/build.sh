#!/usr/bin/env bash
# Builds the SAPIEN CartPole environment into a PufferLib 5.0 native trainer.
#
#   examples/pufferlib5/build.sh PUFFERLIB_DIR [OUT] [--float]
#
# PUFFERLIB_DIR: a checkout of PufferLib's 5.0 branch at commit ea7f989, the commit
# pufferlib-env-capturable.patch targets. OUT (default puffer_sapien_cartpole) is written into it.
# $PYTHON (default python3) must import sapien; only its include and library directories are used.
# Remaining arguments go to PufferLib's build.sh (e.g. --float).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 PUFFERLIB_DIR [OUT] [--float]" >&2
  exit 1
fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PUFFER=$(cd "$1" && pwd)
shift
OUT=puffer_sapien_cartpole
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  OUT=$1
  shift
fi

SAPIEN=$("${PYTHON:-python3}" -c 'import pathlib, sapien; print(pathlib.Path(sapien.__file__).resolve().parent)')
LIBS=$(cd "$SAPIEN/../sapien.libs" && pwd)
BUILD="$PUFFER/build/sapien_cartpole"
mkdir -p "$BUILD"

# The C interface over SAPIEN, built with the package's own directories only.
g++ -std=c++20 -O2 -fPIC -shared \
  -I"$SAPIEN/include" -I"$SAPIEN/include/physx/include" \
  "$HERE/shim/sapien_cartpole.cpp" -o "$BUILD/libsapien_cartpole.so" \
  -L"$LIBS" -lsapien -Wl,-rpath,"$LIBS"

# The environment and its configuration inside the PufferLib tree.
mkdir -p "$PUFFER/ocean/sapien_cartpole"
cp "$HERE/env/sapien_cartpole/sapien_cartpole.cu" "$HERE/env/sapien_cartpole/sapien_cartpole_task.cuh" \
  "$HERE/shim/sapien_cartpole.h" "$PUFFER/ocean/sapien_cartpole/"
cp "$HERE/env/sapien_cartpole.ini" "$PUFFER/config/"

# Graph only the policy and training steps of environments that cannot be recorded (once).
if ! grep -q PUF_ENV_CAPTURABLE "$PUFFER/src/pufferl.cu"; then
  git -C "$PUFFER" apply "$HERE/pufferlib-env-capturable.patch"
fi

# PufferLib splits NVCC_EXTRA on whitespace: the paths must not contain spaces.
cd "$PUFFER"
NVCC_EXTRA="-L$BUILD -lsapien_cartpole -Xlinker -rpath -Xlinker $BUILD" \
  ./build.sh sapien_cartpole "$OUT" --cu "$@"
