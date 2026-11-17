#!/usr/bin/env bash
# Builds a SAPIEN environment into a PufferLib 5.0 native trainer.
#
#   examples/pufferlib5/build.sh SAPIEN_SDK PUFFERLIB_DIR ENV [OUT] [--float]
#
# SAPIEN_SDK: SAPIEN's C++ SDK, an unpacked sapien-sdk-*-linux_x86_64.tar.gz; only its include and
# library directories are used. PUFFERLIB_DIR: a checkout of PufferLib's 5.0 branch at commit
# ea7f989, the commit pufferlib-env-capturable.patch targets. ENV: sapien_cartpole or
# sapien_robot_arm. OUT (default puffer_ENV) is written into PUFFERLIB_DIR.
# Remaining arguments go to PufferLib's build.sh (e.g. --float).
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 SAPIEN_SDK PUFFERLIB_DIR ENV [OUT] [--float]" >&2
  exit 1
fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SDK=$(cd "$1" && pwd)
PUFFER=$(cd "$2" && pwd)
ENV=$3
shift 3
if [ ! -f "$SDK/lib/libsapien.so" ]; then
  echo "not a SAPIEN C++ SDK: $SDK/lib/libsapien.so does not exist" >&2
  exit 1
fi
if [ ! -f "$HERE/shim/$ENV.cpp" ]; then
  echo "unknown environment $ENV: expected $HERE/shim/$ENV.cpp" >&2
  exit 1
fi
OUT=puffer_$ENV
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  OUT=$1
  shift
fi

BUILD="$PUFFER/build/$ENV"
mkdir -p "$BUILD"

# The C interface over SAPIEN, built with the SDK's own directories only.
g++ -std=c++20 -O2 -fPIC -shared \
  -I"$SDK/include" -I"$SDK/include/physx/include" \
  "$HERE/shim/$ENV.cpp" -o "$BUILD/lib$ENV.so" \
  -L"$SDK/lib" -lsapien -Wl,-rpath,"$SDK/lib"

# The environment and its configuration inside the PufferLib tree.
mkdir -p "$PUFFER/ocean/$ENV"
cp "$HERE/env/$ENV/"* "$HERE/shim/$ENV.h" "$PUFFER/ocean/$ENV/"
cp "$HERE/env/$ENV.ini" "$PUFFER/config/"

# Graph only the policy and training steps of environments that cannot be recorded (once).
if ! grep -q PUF_ENV_CAPTURABLE "$PUFFER/src/pufferl.cu"; then
  git -C "$PUFFER" apply "$HERE/pufferlib-env-capturable.patch"
fi

# PufferLib splits NVCC_EXTRA on whitespace: the paths must not contain spaces.
cd "$PUFFER"
NVCC_EXTRA="-L$BUILD -l$ENV -Xlinker -rpath -Xlinker $BUILD" \
  ./build.sh "$ENV" "$OUT" --cu "$@"
